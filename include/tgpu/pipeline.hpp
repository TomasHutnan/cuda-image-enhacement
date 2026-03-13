#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tgpu/image.hpp"

namespace tgpu {

struct NonLocalMeansOptions {
    bool enabled = true;
    float filter_strength = 7.0F / 255.0F;
    int patch_size = 3;
    int search_radius = 1;
};

struct StageExecutionOptions {
    bool non_local_means = true;
    bool unsharp_mask = true;
    bool richardson_lucy = true;
    bool histogram_stretch = true;
};

struct PipelineStage {
    std::uint32_t prefix = 0;
    std::string name;
    ImageF32 image;
};

struct PipelineRunOptions {
    bool capture_intermediate_stages = false;
    NonLocalMeansOptions non_local_means{};
    StageExecutionOptions stage_execution{};
};

struct PipelineRunResult {
    ImageF32 output;
    std::vector<PipelineStage> stages;
};

PipelineRunResult run_pipeline(const ImageGray& input, const PipelineRunOptions& options = {});
PipelineRunResult run_pipeline(const ImageF32& input, const PipelineRunOptions& options = {});

}  // namespace tgpu
