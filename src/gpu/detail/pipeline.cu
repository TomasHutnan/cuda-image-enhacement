// Owns GPU pipeline orchestration, stage sequencing, and generic passthrough helpers.

#include "detail/runtime.hpp"
#include "detail/pipeline_api.hpp"
#include "detail/stage_definition.hpp"

#include <chrono>
#include <optional>

namespace tgpu
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        template <typename Func>
        double timed_ms(Func &&func)
        {
            const auto begin = Clock::now();
            func();
            const auto end = Clock::now();
            return std::chrono::duration<double, std::milli>(end - begin).count();
        }

        template <typename Func>
        double timed_stage_ms(Func &&func, const char *operation)
        {
            const auto begin = Clock::now();
            func();
            throw_if_cuda_failed(cudaDeviceSynchronize(), operation);
            const auto end = Clock::now();
            return std::chrono::duration<double, std::milli>(end - begin).count();
        }

        struct BatchPipelineContext
        {
            bool active = false;
            int width = 0;
            int height = 0;
            std::optional<DevicePipeline> pipeline;
        };

        BatchPipelineContext &batch_pipeline_context()
        {
            static BatchPipelineContext context;
            return context;
        }

        struct StrictKernelSyncGuard
        {
            bool previous = false;

            explicit StrictKernelSyncGuard(bool enabled)
            {
                previous = strict_kernel_sync_checks_enabled();
                set_strict_kernel_sync_checks(enabled);
            }

            ~StrictKernelSyncGuard()
            {
                set_strict_kernel_sync_checks(previous);
            }
        };

        __global__ void passthrough_stage_kernel(const float *input, float *output, std::size_t count)
        {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                                      static_cast<std::size_t>(threadIdx.x);
            if (index < count)
            {
                output[index] = input[index];
            }
        }

        // Helper to execute a stage with optional benchmarking and capture
        template <typename StageFunc>
        void execute_stage_with_capture(
            PipelineRunResult &result,
            DevicePipeline &pipeline,
            const PipelineRunOptions &options,
            bool enabled,
            StageFunc &&stage_func,
            double &benchmark_ms_out,
            std::uint32_t stage_prefix,
            std::string_view stage_name,
            const char *sync_operation)
        {
            if (enabled)
            {
                if (options.collect_benchmark)
                {
                    benchmark_ms_out = timed_stage_ms(stage_func, sync_operation);
                }
                else
                {
                    stage_func();
                }
                commit_stage_output(pipeline);
            }
            capture_stage_if_requested(result, options, pipeline, stage_prefix, stage_name, pipeline.current);
        }

        PipelineRunResult finalize_pipeline(DevicePipeline &pipeline, const PipelineRunOptions &options)
        {
            PipelineRunResult result;
            if (options.collect_benchmark)
            {
                result.benchmark.collected = true;
            }

            capture_stage_if_requested(result, options, pipeline, 0, "input_normalized", pipeline.current);

            // Execute Non-Local Means stage
            execute_stage_with_capture(
                result, pipeline, options,
                options.stage_execution.non_local_means,
                [&]() { run_non_local_means_stage(stage_workspace(pipeline), options.non_local_means); },
                result.benchmark.non_local_means_ms,
                gpu::detail::kPipelineStages[0].capture_prefix, gpu::detail::kPipelineStages[0].name,
                "benchmark non_local_means synchronize");

            // Execute Unsharp Mask stage
            execute_stage_with_capture(
                result, pipeline, options,
                options.stage_execution.unsharp_mask,
                [&]() { run_unsharp_mask_stage(stage_workspace(pipeline), options.unsharp_mask); },
                result.benchmark.unsharp_mask_ms,
                gpu::detail::kPipelineStages[1].capture_prefix, gpu::detail::kPipelineStages[1].name,
                "benchmark unsharp_mask synchronize");

            // Execute Richardson-Lucy stage
            execute_stage_with_capture(
                result, pipeline, options,
                options.stage_execution.richardson_lucy,
                [&]() { run_richardson_lucy_stage(stage_workspace(pipeline), options.richardson_lucy); },
                result.benchmark.richardson_lucy_ms,
                gpu::detail::kPipelineStages[2].capture_prefix, gpu::detail::kPipelineStages[2].name,
                "benchmark richardson_lucy synchronize");

            // Execute Histogram Stretch stage
            execute_stage_with_capture(
                result, pipeline, options,
                options.stage_execution.histogram_stretch,
                [&]() { run_histogram_stretch_stage(stage_workspace(pipeline), options.histogram_stretch); },
                result.benchmark.histogram_stretch_ms,
                gpu::detail::kPipelineStages[3].capture_prefix, gpu::detail::kPipelineStages[3].name,
                "benchmark histogram_stretch synchronize");

            // Download result
            if (options.collect_benchmark)
            {
                result.benchmark.device_to_host_ms = timed_ms([&]() {
                    result.output = download_visible_region(pipeline, pipeline.current);
                });
            }
            else
            {
                result.output = download_visible_region(pipeline, pipeline.current);
            }
            if (options.capture_intermediate_stages)
            {
                result.stages.push_back(PipelineStage{90, "output", result.output});
            }

            return result;
        }
    } // namespace

    void run_passthrough_stage(const StageWorkspace &workspace, const char *operation)
    {
        const std::size_t element_count =
            static_cast<std::size_t>(workspace.expanded_width) * static_cast<std::size_t>(workspace.expanded_height);
        const int block_count = static_cast<int>((element_count + static_cast<std::size_t>(kThreadsPerBlock) - 1) /
                                                 static_cast<std::size_t>(kThreadsPerBlock));

        passthrough_stage_kernel<<<block_count, kThreadsPerBlock>>>(workspace.input, workspace.output, element_count);
        throw_if_kernel_failed(operation);
    }

    PipelineRunResult run_pipeline_cuda(const ImageGray &input, const PipelineRunOptions &options)
    {
        if (input.empty())
        {
            return {};
        }

        StrictKernelSyncGuard sync_guard(options.strict_kernel_sync_checks);

        BatchPipelineContext &context = batch_pipeline_context();
        DevicePipeline local_pipeline;
        DevicePipeline *pipeline = nullptr;
        if (context.active)
        {
            if (context.width != input.width || context.height != input.height)
            {
                throw std::runtime_error("run_pipeline: image dimensions differ from active batch context");
            }
            if (!context.pipeline.has_value())
            {
                context.pipeline.emplace(create_pipeline(input.width, input.height));
            }
            pipeline = &context.pipeline.value();
        }
        else
        {
            local_pipeline = create_pipeline(input.width, input.height);
            pipeline = &local_pipeline;
        }

        PipelineRunResult result;
        if (options.collect_benchmark)
        {
            result.benchmark.collected = true;
            result.benchmark.host_to_device_ms = timed_ms([&]() {
                initialize_pipeline_from_raw(input, *pipeline);
            });
            PipelineRunResult tail = finalize_pipeline(*pipeline, options);
            tail.benchmark.host_to_device_ms = result.benchmark.host_to_device_ms;
            return tail;
        }

        initialize_pipeline_from_raw(input, *pipeline);
        return finalize_pipeline(*pipeline, options);
    }

    PipelineRunResult run_pipeline_cuda(const ImageF32 &input, const PipelineRunOptions &options)
    {
        if (input.empty())
        {
            return {};
        }

        StrictKernelSyncGuard sync_guard(options.strict_kernel_sync_checks);

        BatchPipelineContext &context = batch_pipeline_context();
        DevicePipeline local_pipeline;
        DevicePipeline *pipeline = nullptr;
        if (context.active)
        {
            if (context.width != input.width || context.height != input.height)
            {
                throw std::runtime_error("run_pipeline: image dimensions differ from active batch context");
            }
            if (!context.pipeline.has_value())
            {
                context.pipeline.emplace(create_pipeline(input.width, input.height));
            }
            pipeline = &context.pipeline.value();
        }
        else
        {
            local_pipeline = create_pipeline(input.width, input.height);
            pipeline = &local_pipeline;
        }

        PipelineRunResult result;
        if (options.collect_benchmark)
        {
            result.benchmark.collected = true;
            result.benchmark.host_to_device_ms = timed_ms([&]() {
                initialize_pipeline_from_normalized(input, *pipeline);
            });
            PipelineRunResult tail = finalize_pipeline(*pipeline, options);
            tail.benchmark.host_to_device_ms = result.benchmark.host_to_device_ms;
            return tail;
        }

        initialize_pipeline_from_normalized(input, *pipeline);
        return finalize_pipeline(*pipeline, options);
    }

    void begin_pipeline_batch_cuda(int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw std::runtime_error("begin_pipeline_batch: width and height must be positive");
        }

        BatchPipelineContext &context = batch_pipeline_context();
        const bool same_shape = context.active && context.width == width && context.height == height;
        if (same_shape)
        {
            return;
        }

        context.pipeline.reset();
        context.pipeline.emplace(create_pipeline(width, height));
        context.active = true;
        context.width = width;
        context.height = height;
    }

    void end_pipeline_batch_cuda()
    {
        BatchPipelineContext &context = batch_pipeline_context();
        context.pipeline.reset();
        context.active = false;
        context.width = 0;
        context.height = 0;
    }
} // namespace tgpu