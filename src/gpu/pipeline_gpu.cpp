// Public pipeline entrypoints that forward into the CUDA implementation.

#include "tgpu/pipeline.hpp"

namespace tgpu
{
    PipelineRunResult run_pipeline_cuda(const ImageGray &input, const PipelineRunOptions &options);
    PipelineRunResult run_pipeline_cuda(const ImageF32 &input, const PipelineRunOptions &options);

    PipelineRunResult run_pipeline(const ImageGray &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda(input, options);
    }

    PipelineRunResult run_pipeline(const ImageF32 &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda(input, options);
    }
} // namespace tgpu
