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

struct HistogramStretchOptions {
    bool enabled = true;
    float saturation_percent = 0.5F;
    int histogram_bins = 4096;
};

struct UnsharpMaskOptions {
    float sigma = 1.6666667F;
    float amount = 0.6F;
};

struct RichardsonLucyOptions {
    int iterations = 2;
    float psf_sigma = 2.5F;
    int psf_radius = 7;
    float epsilon = 1.0e-7F;
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

struct PipelineBenchmark {
    bool collected = false;
    double host_to_device_ms = 0.0;
    double non_local_means_ms = 0.0;
    double unsharp_mask_ms = 0.0;
    double richardson_lucy_ms = 0.0;
    double histogram_stretch_ms = 0.0;
    double device_to_host_ms = 0.0;
};

struct PipelineRunOptions {
    bool capture_intermediate_stages = false;
    bool collect_benchmark = false;
    NonLocalMeansOptions non_local_means{};
    UnsharpMaskOptions unsharp_mask{};
    RichardsonLucyOptions richardson_lucy{};
    HistogramStretchOptions histogram_stretch{};
    StageExecutionOptions stage_execution{};
};

struct PipelineRunResult {
    ImageF32 output;
    std::vector<PipelineStage> stages;
    PipelineBenchmark benchmark{};
};

PipelineRunResult run_pipeline(const ImageGray& input, const PipelineRunOptions& options = {});
PipelineRunResult run_pipeline(const ImageF32& input, const PipelineRunOptions& options = {});

}  // namespace tgpu
