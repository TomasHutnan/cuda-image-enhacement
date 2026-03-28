// Owns GPU pipeline orchestration, stage sequencing, and generic passthrough helpers.

#include "detail/runtime.hpp"

#include <chrono>

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

        struct StageDefinition
        {
            std::uint32_t prefix;
            std::string_view name;
        };

        constexpr StageDefinition kPipelineStages[] = {
            {10, "non_local_means"},
            {20, "unsharp_mask"},
            {30, "richardson_lucy"},
            {40, "histogram_stretch"},
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

        PipelineRunResult finalize_pipeline(DevicePipeline &pipeline, const PipelineRunOptions &options)
        {
            PipelineRunResult result;
            if (options.collect_benchmark)
            {
                result.benchmark.collected = true;
            }

            capture_stage_if_requested(result, options, pipeline, 0, "input_normalized", pipeline.current);

            if (options.stage_execution.non_local_means)
            {
                if (options.collect_benchmark)
                {
                    result.benchmark.non_local_means_ms = timed_ms([&]() {
                        run_non_local_means_stage(stage_workspace(pipeline), options.non_local_means);
                    });
                }
                else
                {
                    run_non_local_means_stage(stage_workspace(pipeline), options.non_local_means);
                }
                commit_stage_output(pipeline);
            }
            capture_stage_if_requested(result, options, pipeline, kPipelineStages[0].prefix, kPipelineStages[0].name, pipeline.current);

            if (options.stage_execution.unsharp_mask)
            {
                if (options.collect_benchmark)
                {
                    result.benchmark.unsharp_mask_ms = timed_ms([&]() {
                        run_unsharp_mask_stage(stage_workspace(pipeline), options.unsharp_mask);
                    });
                }
                else
                {
                    run_unsharp_mask_stage(stage_workspace(pipeline), options.unsharp_mask);
                }
                commit_stage_output(pipeline);
            }
            capture_stage_if_requested(result, options, pipeline, kPipelineStages[1].prefix, kPipelineStages[1].name, pipeline.current);

            if (options.stage_execution.richardson_lucy)
            {
                if (options.collect_benchmark)
                {
                    result.benchmark.richardson_lucy_ms = timed_ms([&]() {
                        run_richardson_lucy_stage(stage_workspace(pipeline), options.richardson_lucy);
                    });
                }
                else
                {
                    run_richardson_lucy_stage(stage_workspace(pipeline), options.richardson_lucy);
                }
                commit_stage_output(pipeline);
            }
            capture_stage_if_requested(result, options, pipeline, kPipelineStages[2].prefix, kPipelineStages[2].name, pipeline.current);

            if (options.stage_execution.histogram_stretch)
            {
                if (options.collect_benchmark)
                {
                    result.benchmark.histogram_stretch_ms = timed_ms([&]() {
                        run_histogram_stretch_stage(stage_workspace(pipeline), options.histogram_stretch);
                    });
                }
                else
                {
                    run_histogram_stretch_stage(stage_workspace(pipeline), options.histogram_stretch);
                }
                commit_stage_output(pipeline);
            }
            capture_stage_if_requested(result, options, pipeline, kPipelineStages[3].prefix, kPipelineStages[3].name, pipeline.current);

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

        DevicePipeline pipeline = create_pipeline(input.width, input.height);
        PipelineRunResult result;
        if (options.collect_benchmark)
        {
            result.benchmark.collected = true;
            result.benchmark.host_to_device_ms = timed_ms([&]() {
                initialize_pipeline_from_raw(input, pipeline);
            });
            PipelineRunResult tail = finalize_pipeline(pipeline, options);
            tail.benchmark.host_to_device_ms = result.benchmark.host_to_device_ms;
            return tail;
        }

        initialize_pipeline_from_raw(input, pipeline);
        return finalize_pipeline(pipeline, options);
    }

    PipelineRunResult run_pipeline_cuda(const ImageF32 &input, const PipelineRunOptions &options)
    {
        if (input.empty())
        {
            return {};
        }

        DevicePipeline pipeline = create_pipeline(input.width, input.height);
        PipelineRunResult result;
        if (options.collect_benchmark)
        {
            result.benchmark.collected = true;
            result.benchmark.host_to_device_ms = timed_ms([&]() {
                initialize_pipeline_from_normalized(input, pipeline);
            });
            PipelineRunResult tail = finalize_pipeline(pipeline, options);
            tail.benchmark.host_to_device_ms = result.benchmark.host_to_device_ms;
            return tail;
        }

        initialize_pipeline_from_normalized(input, pipeline);
        return finalize_pipeline(pipeline, options);
    }
} // namespace tgpu