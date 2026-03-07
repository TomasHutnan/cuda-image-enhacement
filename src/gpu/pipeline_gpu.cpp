#include "tgpu/pipeline.hpp"

#include <utility>

namespace tgpu {

PipelineRunResult run_pipeline(const ImageF32& input, const PipelineRunOptions& options) {
    PipelineRunResult result;

    if (options.capture_intermediate_stages) {
        result.stages.push_back(PipelineStage{"input_normalized", input});
    }

    // The actual GPU stages will replace this passthrough as they are implemented.
    result.output = input;

    if (options.capture_intermediate_stages) {
        result.stages.push_back(PipelineStage{"output", result.output});
    }

    return result;
}

}  // namespace tgpu
