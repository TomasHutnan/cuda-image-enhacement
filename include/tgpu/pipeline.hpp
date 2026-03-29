#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "tgpu/image.hpp"

namespace tgpu {

/// Non-Local Means denoising options.
/// Patch-based adaptive denoising algorithm.
struct NonLocalMeansOptions {
    /// Enable/disable this stage in the pipeline
    bool enabled = true;
    /// Filter strength [0.0-1.0]; higher = more aggressive denoising; typical: 7/255 ≈ 0.027
    float filter_strength = 7.0F / 255.0F;
    /// Patch size in pixels; must be odd [1, 9]; larger = slower but smoother results
    int patch_size = 3;
    /// Search radius in pixels [1, 5]; larger = slower, more thorough; typical: 1-2
    int search_radius = 1;
};

/// Histogram stretch (percentile-based contrast enhancement) options.
/// Stretches intensity range based on visible image percentiles.
struct HistogramStretchOptions {
    /// Enable/disable this stage in the pipeline
    bool enabled = true;
    /// Saturation percentage [0.0-50.0]; typical: 0.5; % of dark/bright pixels to clip
    float saturation_percent = 0.5F;
    /// Histogram bin count [256, 65536]; typical: 4096; higher = finer precision, more memory
    int bin_count = 4096;
};

/// Unsharp mask (Gaussian blur + enhancement) options.
/// Enhances edges by subtracting blurred image from original.
struct UnsharpMaskOptions {
    /// Enable/disable this stage in the pipeline
    bool enabled = true;
    /// Gaussian standard deviation [0.5, 5.0]; typical: 1.67; affects blur radius
    float sigma = 1.6666667F;
    /// Blend amount [0.0-1.0]; typical: 0.6; higher = stronger enhancement
    float amount = 0.6F;
};

/// Richardson-Lucy iterative deconvolution options.
/// Blind deconvolution with Gaussian PSF kernel.
struct RichardsonLucyOptions {
    /// Enable/disable this stage in the pipeline
    bool enabled = true;
    /// Number of iterations [1, 10]; typical: 2-5; higher = more deconvolution but risk of artifacts
    int iterations = 2;
    /// PSF Gaussian standard deviation [0.5, 5.0]; typical: 2.5; width of assumed blur
    float psf_sigma = 2.5F;
    /// PSF kernel half-width in pixels [1, 15]; typical: 7; derived from psf_sigma
    int psf_radius = 7;
    /// Regularization epsilon [1e-9, 1e-5]; prevents division by zero; typical: 1e-7
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
    bool strict_kernel_sync_checks = false;
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

// Initializes persistent GPU workspace for processing a same-resolution batch.
void begin_pipeline_batch(int width, int height);
// Releases persistent GPU workspace allocated by begin_pipeline_batch.
void end_pipeline_batch();

}  // namespace tgpu
