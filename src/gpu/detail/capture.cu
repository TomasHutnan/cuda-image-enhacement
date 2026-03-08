// Downloads visible output regions and optionally captures named pipeline stages.

#include "detail/runtime.hpp"

namespace tgpu
{
    ImageF32 download_visible_region(const DevicePipeline &pipeline, const float *source)
    {
        ImageF32 image;
        image.width = pipeline.width;
        image.height = pipeline.height;
        image.stride = pipeline.width;
        image.data.resize(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height));

        const float *visible_source = source +
                                      static_cast<std::size_t>(pipeline.border) * static_cast<std::size_t>(pipeline.expanded_width) +
                                      static_cast<std::size_t>(pipeline.border);
        throw_if_cuda_failed(
            cudaMemcpy2D(
                image.data.data(),
                static_cast<std::size_t>(image.stride) * sizeof(float),
                visible_source,
                static_cast<std::size_t>(pipeline.expanded_width) * sizeof(float),
                static_cast<std::size_t>(image.width) * sizeof(float),
                static_cast<std::size_t>(image.height),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy2D download visible image");
        return image;
    }

    void capture_stage_if_requested(
        PipelineRunResult &result,
        const PipelineRunOptions &options,
        const DevicePipeline &pipeline,
        std::uint32_t prefix,
        std::string_view name,
        const float *source)
    {
        if (!options.capture_intermediate_stages)
        {
            return;
        }

        result.stages.push_back(PipelineStage{prefix, std::string{name}, download_visible_region(pipeline, source)});
    }
} // namespace tgpu