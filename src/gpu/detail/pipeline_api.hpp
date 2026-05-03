#pragma once

// Internal declarations shared across GPU detail translation units.

#include "detail/buffers.hpp"

#include <string_view>

namespace tgpu
{
    DeviceFloatBuffer allocate_float_buffer(std::size_t element_count);
    DeviceByteBuffer upload_raw_image(const ImageGray &input);
    DeviceFloatBuffer upload_normalized_image(const ImageF32 &input);

    DevicePipeline create_pipeline(int width, int height);
    void initialize_pipeline_from_raw(const ImageGray &input, DevicePipeline &pipeline);
    void initialize_pipeline_from_normalized(const ImageF32 &input, DevicePipeline &pipeline);

    void run_non_local_means_stage(const StageWorkspace &workspace, const NonLocalMeansOptions &options);
    void run_unsharp_mask_stage(const StageWorkspace &workspace, const UnsharpMaskOptions &options);
    void run_richardson_lucy_stage(const StageWorkspace &workspace, const RichardsonLucyOptions &options);
    void run_histogram_stretch_stage(const StageWorkspace &workspace, const HistogramStretchOptions &options);

    ImageF32 download_visible_region(const DevicePipeline &pipeline, const float *source);
    DeviceImageF32 download_visible_region_to_device(const DevicePipeline &pipeline, const float *source);
    void capture_stage_if_requested(
        PipelineRunResult &result,
        const PipelineRunOptions &options,
        const DevicePipeline &pipeline,
        std::uint32_t prefix,
        std::string_view name,
        const float *source);

    PipelineRunResult run_pipeline_cuda(const ImageGray &input, const PipelineRunOptions &options);
    PipelineRunResult run_pipeline_cuda(const ImageF32 &input, const PipelineRunOptions &options);
    PipelineRunDeviceResult run_pipeline_cuda_device(const ImageGray &input, const PipelineRunOptions &options);
    PipelineRunDeviceResult run_pipeline_cuda_device(const ImageF32 &input, const PipelineRunOptions &options);
    void begin_pipeline_batch_cuda(int width, int height);
    void end_pipeline_batch_cuda();

} // namespace tgpu