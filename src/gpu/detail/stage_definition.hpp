#pragma once

#include <string_view>
#include <array>

namespace tgpu::gpu::detail {

/// Metadata for a GPU processing stage.
struct StageMetadata {
    std::string_view name;           ///< Stage name (e.g., "non_local_means")
    std::string_view description;    ///< Algorithm description
    bool enabled_by_default;         ///< Whether enabled in default pipeline
    std::string_view validation_hint; ///< Hint for parameter validation errors
};

/// Registry of all GPU processing stages.
constexpr std::array<StageMetadata, 4> kPiplelineStages{{
    {
        "non_local_means",
        "Patch-based denoising with adaptive filtering",
        true,
        "patch_size must be odd and >= 1; search_radius must be >= 1"
    },
    {
        "unsharp_mask",
        "Gaussian blur + contrast enhancement",
        true,
        "sigma must be > 0.0; amount should be non-zero"
    },
    {
        "richardson_lucy",
        "Iterative blind deconvolution with Gaussian PSF",
        true,
        "iterations must be >= 1; psf_sigma > 0.0; psf_radius > 0; epsilon > 0.0"
    },
    {
        "histogram_stretch",
        "Percentile-based contrast stretching",
        true,
        "saturation_percent in (0, 50); bin_count >= 256"
    }
}};

/// Look up stage metadata by name.
/// @param stage_name Stage identifier
/// @return Pointer to metadata, or nullptr if not found
inline const StageMetadata* find_stage_metadata(std::string_view stage_name) {
    for (const auto& stage : kPiplelineStages) {
        if (stage.name == stage_name) {
            return &stage;
        }
    }
    return nullptr;
}

}  // namespace tgpu::gpu::detail
