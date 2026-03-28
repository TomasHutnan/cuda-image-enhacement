// Public pipeline entrypoints that forward into the CUDA implementation.

#include "tgpu/pipeline.hpp"

namespace tgpu
{
    PipelineRunResult run_pipeline_cuda(const ImageGray &input, const PipelineRunOptions &options);
    PipelineRunResult run_pipeline_cuda(const ImageF32 &input, const PipelineRunOptions &options);
    void begin_pipeline_batch_cuda(int width, int height);
    void end_pipeline_batch_cuda();

    PipelineRunResult run_pipeline(const ImageGray &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda(input, options);
    }

    PipelineRunResult run_pipeline(const ImageF32 &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda(input, options);
    }

    void begin_pipeline_batch(int width, int height)
    {
        begin_pipeline_batch_cuda(width, height);
    }

    void end_pipeline_batch()
    {
        end_pipeline_batch_cuda();
    }
} // namespace tgpu
