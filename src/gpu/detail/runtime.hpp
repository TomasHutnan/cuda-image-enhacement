#pragma once

// Private CUDA pipeline helpers and cross-file function declarations.

#include "detail/types.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tgpu
{
    inline void throw_if_cuda_failed(cudaError_t status, const char *operation)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
    }

    inline void throw_if_kernel_failed(const char *operation)
    {
        throw_if_cuda_failed(cudaGetLastError(), operation);
        throw_if_cuda_failed(cudaDeviceSynchronize(), operation);
    }

    inline int compute_border(int width, int height)
    {
        return std::max(width / 5, height / 5);
    }

    inline StageWorkspace stage_workspace(const DevicePipeline &pipeline)
    {
        return StageWorkspace{
            .input = pipeline.current,
            .output = pipeline.next,
            .auxiliary = pipeline.scratch,
            .width = pipeline.width,
            .height = pipeline.height,
            .border = pipeline.border,
            .expanded_width = pipeline.expanded_width,
            .expanded_height = pipeline.expanded_height,
        };
    }

    inline void commit_stage_output(DevicePipeline &pipeline)
    {
        float *previous_input = pipeline.current;
        pipeline.current = pipeline.next;
        pipeline.next = previous_input;
    }

    DeviceFloatBuffer allocate_float_buffer(std::size_t element_count);
    DeviceByteBuffer upload_raw_image(const ImageGray &input);
    DeviceFloatBuffer upload_normalized_image(const ImageF32 &input);

    DevicePipeline create_pipeline(int width, int height);
    void initialize_pipeline_from_raw(const ImageGray &input, DevicePipeline &pipeline);
    void initialize_pipeline_from_normalized(const ImageF32 &input, DevicePipeline &pipeline);

    void run_non_local_means_stage(const StageWorkspace &workspace);
    void run_unsharp_mask_stage(const StageWorkspace &workspace);
    void run_richardson_lucy_stage(const StageWorkspace &workspace);
    void run_histogram_stretch_stage(const StageWorkspace &workspace);
    void run_passthrough_stage(const StageWorkspace &workspace, const char *operation);

    ImageF32 download_visible_region(const DevicePipeline &pipeline, const float *source);
    void capture_stage_if_requested(
        PipelineRunResult &result,
        const PipelineRunOptions &options,
        const DevicePipeline &pipeline,
        std::uint32_t prefix,
        std::string_view name,
        const float *source);

    PipelineRunResult run_pipeline_cuda(const ImageGray &input, const PipelineRunOptions &options);
    PipelineRunResult run_pipeline_cuda(const ImageF32 &input, const PipelineRunOptions &options);

} // namespace tgpu