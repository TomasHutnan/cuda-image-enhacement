// Contains the non-local means stage implementation and stage-local kernels.

#include "detail/runtime.hpp"
#include "detail/compute.hpp"

#include <cmath>

namespace tgpu
{
    namespace
    {
        /// Early rejection threshold exponent: skips patch computation if center pixel
        /// difference exceeds kEarlyRejectExponent * filter_strength^2.
        /// exp(-12) ≈ 6e-6, negligible weight; safe optimization for efficiency.
        constexpr float kEarlyRejectExponent = 12.0F;

        /// Patch-based non-local denoising kernel.
        /// For each pixel, searches nearby pixels for similar patches and computes
        /// weighted average based on patch similarity. Early rejection on center-pixel
        /// distance saves computation for dissimilar neighbors.
        ///
        /// Algorithm:
        /// 1. For each pixel at (x,y):
        /// 2.   For each candidate neighbor in [-search_radius, +search_radius]^2:
        /// 3.     Compute center-pixel distance; skip if > early_reject_distance (early rejection)
        /// 4.     Compute patch distance (sum of squared differences in patch region)
        /// 5.     Weight = exp(-patch_distance / sigma^2)
        /// 6.   Output = sum(weight * neighbor) / sum(weight) with self-weight adjustment
        ///
        /// Filter strength (sigma) controls noise assumption; higher = more aggressive denoising.
        __global__ void non_local_means_kernel(
            const float *input,
            float *output,
            int expanded_width,
            int expanded_height,
            int patch_radius,
            int search_radius,
            float filter_strength)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            const std::size_t center_index = expanded_index(x, y, expanded_width);
            const float center_value = input[center_index];
            const float filter_strength_squared = filter_strength * filter_strength;
            const float early_reject_distance = kEarlyRejectExponent * filter_strength_squared;

            float weighted_sum = 0.0F;
            float total_weight = 0.0F;
            float self_weight = 0.0F;

            for (int offset_y = -search_radius; offset_y <= search_radius; ++offset_y)
            {
                for (int offset_x = -search_radius; offset_x <= search_radius; ++offset_x)
                {
                    if (offset_x == 0 && offset_y == 0)
                    {
                        continue;
                    }

                    const int neighbor_x = reflect_coordinate(x + offset_x, expanded_width);
                    const int neighbor_y = reflect_coordinate(y + offset_y, expanded_height);
                    const std::size_t neighbor_index = expanded_index(neighbor_x, neighbor_y, expanded_width);
                    const float neighbor_value = input[neighbor_index];

                    // Early rejection: skip if center pixels are too different
                    const float center_difference = center_value - neighbor_value;
                    const float center_distance = center_difference * center_difference;
                    if (center_distance > early_reject_distance)
                    {
                        continue;
                    }

                    // Compute full patch distance
                    float patch_distance = 0.0F;
                    for (int patch_y = -patch_radius; patch_y <= patch_radius; ++patch_y)
                    {
                        for (int patch_x = -patch_radius; patch_x <= patch_radius; ++patch_x)
                        {
                            const int sample_x = reflect_coordinate(x + patch_x, expanded_width);
                            const int sample_y = reflect_coordinate(y + patch_y, expanded_height);
                            const int neighbor_sample_x = reflect_coordinate(neighbor_x + patch_x, expanded_width);
                            const int neighbor_sample_y = reflect_coordinate(neighbor_y + patch_y, expanded_height);

                            const float difference =
                                input[expanded_index(sample_x, sample_y, expanded_width)] -
                                input[expanded_index(neighbor_sample_x, neighbor_sample_y, expanded_width)];
                            patch_distance += difference * difference;
                        }
                    }

                    const float weight = expf(-patch_distance / filter_strength_squared);
                    self_weight = weight > self_weight ? weight : self_weight;
                    weighted_sum += weight * neighbor_value;
                    total_weight += weight;
                }
            }

            weighted_sum += self_weight * center_value;
            total_weight += self_weight;

            output[expanded_index(x, y, expanded_width)] = total_weight > 0.0F ? weighted_sum / total_weight : center_value;
        }
    }  // namespace

    void run_non_local_means_stage(const StageWorkspace &workspace, const NonLocalMeansOptions &options)
    {
        if (!options.enabled)
        {
            run_passthrough_stage(workspace, "non-local means disabled");
            return;
        }

        // Validate parameters
        if (options.patch_size < 1 || options.search_radius < 0 || options.filter_strength <= 0.0F)
        {
            run_passthrough_stage(workspace, "NLM: invalid parameters (patch_size >= 1, search_radius >= 0, filter_strength > 0)");
            return;
        }

        // Ensure patch_size is odd
        int patch_size = max(options.patch_size, 1);
        if ((patch_size % 2) == 0)
        {
            patch_size += 1;
        }

        const int patch_radius = patch_size / 2;
        const int search_radius = max(options.search_radius, 0);
        const float filter_strength = options.filter_strength > 1.0e-6F ? options.filter_strength : 1.0e-6F;

        // Use standardized grid calculation utility
        const dim3 grid_size = gpu::detail::compute_2d_grid(
            workspace.expanded_width,
            workspace.expanded_height,
            gpu::detail::block_sizes::kImage2D_X,
            gpu::detail::block_sizes::kImage2D_Y);

        non_local_means_kernel<<<grid_size, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                   gpu::detail::block_sizes::kImage2D_Y, 1)>>>(
            workspace.input,
            workspace.output,
            workspace.expanded_width,
            workspace.expanded_height,
            patch_radius,
            search_radius,
            filter_strength);
        throw_if_kernel_failed("non-local means stage");
    }
}  // namespace tgpu