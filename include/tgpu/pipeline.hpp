#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tgpu/image.hpp"

namespace tgpu {

struct PipelineStage {
    std::uint32_t prefix = 0;
    std::string name;
    ImageF32 image;
};

struct PipelineRunOptions {
    bool capture_intermediate_stages = false;
};

struct PipelineRunResult {
    ImageF32 output;
    std::vector<PipelineStage> stages;
};

PipelineRunResult run_pipeline(const ImageF32& input, const PipelineRunOptions& options = {});

}  // namespace tgpu
