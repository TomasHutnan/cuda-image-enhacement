// Unsharp mask (Gaussian blur + enhancement) stage implementation.

#include "detail/runtime.hpp"
#include "detail/kernel_grid.hpp"
#include "detail/kernel_cache.hpp"

#include <array>
#include <cmath>

namespace tgpu
{
    namespace
    {
        /// Maximum Gaussian kernel half-width in pixels.
        /// radius = ceil(3 * sigma); clamped to [1, kMaxGaussianRadius]
        constexpr int kMaxGaussianRadius = 64;
        constexpr int kMaxGaussianKernelSize = 2 * kMaxGaussianRadius + 1;

        __constant__ float kGaussianWeights[kMaxGaussianKernelSize];

        /// Unsharp mask: Gaussian blur + contrast enhancement.
        /// Algorithm:
        /// 1. Compute Gaussian blur (separable: horizontal then vertical)
        /// 2. Compute unsharp enhance: output = original + amount * (original - blurred)
        /// 3. Output clamped to [0.0, 1.0]
        ///
        /// Enhancement amount controls strength; typical: 0.6
        /// Sigma controls blur width; typical: 1.667
        
        // Global cache for Gaussian kernel weights (device constant memory)
        gpu::detail::KernelCache<float, kMaxGaussianKernelSize> g_gaussian_cache;
        
        // Helper to compute Gaussian kernel weights on host
        void compute_gaussian_kernel(int radius, float sigma, std::array<float, kMaxGaussianKernelSize>& weights)
        {
            float weight_sum = 0.0F;
            const int kernel_size = 2 * radius + 1;
            const float sigma_squared = sigma * sigma;

            for (int index = 0; index < kernel_size; ++index)
            {
                const int offset = index - radius;
                const float distance_squared = static_cast<float>(offset * offset);
                const float weight = expf(-distance_squared / (2.0F * sigma_squared));
                weights[static_cast<std::size_t>(index)] = weight;
                weight_sum += weight;
            }

            // Normalize
            if (weight_sum > 0.0F)
            {
                for (int index = 0; index < kernel_size; ++index)
                {
                    weights[static_cast<std::size_t>(index)] /= weight_sum;
                }
            }
        }

        __global__ void gaussian_horizontal_kernel(
            const float *input,
            float *horizontal_blur,
            int expanded_width,
            int expanded_height,
            int radius)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            float blurred = 0.0F;
            for (int offset_x = -radius; offset_x <= radius; ++offset_x)
            {
                const int sample_x = clamp_coordinate(x + offset_x, expanded_width);
                const float weight = kGaussianWeights[offset_x + radius];
                blurred += weight * input[expanded_index(sample_x, y, expanded_width)];
            }

            horizontal_blur[expanded_index(x, y, expanded_width)] = blurred;
        }

        __global__ void gaussian_vertical_unsharp_kernel(
            const float *original,
            const float *horizontal_blur,
            float *output,
            int expanded_width,
            int expanded_height,
            int radius,
            float amount)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            float blurred = 0.0F;
            for (int offset_y = -radius; offset_y <= radius; ++offset_y)
            {
                const int sample_y = clamp_coordinate(y + offset_y, expanded_height);
                const float weight = kGaussianWeights[offset_y + radius];
                blurred += weight * horizontal_blur[expanded_index(x, sample_y, expanded_width)];
            }

            const std::size_t index = expanded_index(x, y, expanded_width);
            const float original_value = original[index];
            const float enhanced = original_value + amount * (original_value - blurred);
            output[index] = fminf(fmaxf(enhanced, 0.0F), 1.0F);
        }
    } // namespace

    void run_unsharp_mask_stage(const StageWorkspace &workspace, const UnsharpMaskOptions &options)
    {
        if (!options.enabled)
        {
            run_passthrough_stage(workspace, "unsharp mask disabled");
            return;
        }

        // Validate parameters
        if (!std::isfinite(options.sigma) || options.sigma <= 0.0F)
        {
            run_passthrough_stage(workspace, "Unsharp: sigma must be > 0.0 and finite");
            return;
        }

        if (!std::isfinite(options.amount) || fabsf(options.amount) <= 1.0e-6F)
        {
            run_passthrough_stage(workspace, "Unsharp: amount must be non-zero and finite");
            return;
        }

        // Compute kernel parameters
        const int radius = std::clamp(static_cast<int>(ceilf(3.0F * options.sigma)), 1, kMaxGaussianRadius);
        const int kernel_size = 2 * radius + 1;

        // Check/update cache
        if (!g_gaussian_cache.is_valid(options.sigma, radius))
        {
            std::array<float, kMaxGaussianKernelSize> weights{};
            compute_gaussian_kernel(radius, options.sigma, weights);

            if (weights[0] <= 0.0F)  // Sanity check (shouldn't happen if normalize succeeded)
            {
                run_passthrough_stage(workspace, "Unsharp: degenerate Gaussian kernel");
                return;
            }

            g_gaussian_cache.update(options.sigma, radius, weights.data(), (void*)kGaussianWeights, kernel_size);
        }

        // Use standardized grid calculation
        const dim3 grid_size = gpu::detail::compute_2d_grid(
            workspace.expanded_width,
            workspace.expanded_height,
            gpu::detail::block_sizes::kImage2D_X,
            gpu::detail::block_sizes::kImage2D_Y);

        // Horizontal blur pass
        gaussian_horizontal_kernel<<<grid_size, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                       gpu::detail::block_sizes::kImage2D_Y, 1)>>>(
            workspace.input,
            workspace.auxiliary,
            workspace.expanded_width,
            workspace.expanded_height,
            radius);
        throw_if_kernel_failed("unsharp mask horizontal gaussian");

        // Vertical blur + enhancement pass
        gaussian_vertical_unsharp_kernel<<<grid_size, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                             gpu::detail::block_sizes::kImage2D_Y, 1)>>>(
            workspace.input,
            workspace.auxiliary,
            workspace.output,
            workspace.expanded_width,
            workspace.expanded_height,
            radius,
            options.amount);
        throw_if_kernel_failed("unsharp mask vertical gaussian + apply");
    }
} // namespace tgpu