// Placeholder for the unsharp mask stage until the CUDA implementation is added.

#include "detail/runtime.hpp"

#include <array>
#include <cmath>

namespace tgpu
{
    namespace
    {
        constexpr int kMaxGaussianRadius = 64;
        constexpr int kMaxGaussianKernelSize = 2 * kMaxGaussianRadius + 1;

        __constant__ float kGaussianWeights[kMaxGaussianKernelSize];

        struct GaussianKernelCache
        {
            bool valid = false;
            float sigma = 0.0F;
            int radius = 0;
            int kernel_size = 0;
            std::array<float, kMaxGaussianKernelSize> weights{};
        };

        GaussianKernelCache &gaussian_kernel_cache()
        {
            static GaussianKernelCache cache;
            return cache;
        }

        bool same_sigma(float lhs, float rhs)
        {
            return fabsf(lhs - rhs) <= 1.0e-6F;
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
        if (!std::isfinite(options.sigma) || options.sigma <= 0.0F)
        {
            run_passthrough_stage(workspace, "unsharp mask invalid sigma");
            return;
        }

        if (!std::isfinite(options.amount))
        {
            run_passthrough_stage(workspace, "unsharp mask invalid amount");
            return;
        }

        if (fabsf(options.amount) <= 1.0e-6F)
        {
            run_passthrough_stage(workspace, "unsharp mask zero amount");
            return;
        }

        const int radius = std::clamp(static_cast<int>(ceilf(3.0F * options.sigma)), 1, kMaxGaussianRadius);
        const int kernel_size = 2 * radius + 1;

        GaussianKernelCache &cache = gaussian_kernel_cache();
        const bool cache_hit =
            cache.valid && cache.radius == radius && cache.kernel_size == kernel_size && same_sigma(cache.sigma, options.sigma);
        if (!cache_hit)
        {
            float weight_sum = 0.0F;
            const float sigma_squared = options.sigma * options.sigma;

            for (int index = 0; index < kernel_size; ++index)
            {
                const int offset = index - radius;
                const float distance_squared = static_cast<float>(offset * offset);
                const float weight = expf(-distance_squared / (2.0F * sigma_squared));
                cache.weights[static_cast<std::size_t>(index)] = weight;
                weight_sum += weight;
            }

            if (weight_sum <= 0.0F)
            {
                run_passthrough_stage(workspace, "unsharp mask degenerate kernel");
                return;
            }

            for (int index = 0; index < kernel_size; ++index)
            {
                cache.weights[static_cast<std::size_t>(index)] /= weight_sum;
            }

            cache.valid = true;
            cache.sigma = options.sigma;
            cache.radius = radius;
            cache.kernel_size = kernel_size;

            throw_if_cuda_failed(
                cudaMemcpyToSymbol(
                    kGaussianWeights,
                    cache.weights.data(),
                    static_cast<std::size_t>(kernel_size) * sizeof(float),
                    0,
                    cudaMemcpyHostToDevice),
                "cudaMemcpyToSymbol unsharp gaussian weights");
        }

        const dim3 grid_size{
            static_cast<unsigned int>((workspace.expanded_width + static_cast<int>(kImageBlockSize.x) - 1) /
                                      static_cast<int>(kImageBlockSize.x)),
            static_cast<unsigned int>((workspace.expanded_height + static_cast<int>(kImageBlockSize.y) - 1) /
                                      static_cast<int>(kImageBlockSize.y)),
            1};

        gaussian_horizontal_kernel<<<grid_size, kImageBlockSize>>>(
            workspace.input,
            workspace.auxiliary,
            workspace.expanded_width,
            workspace.expanded_height,
            radius);
        throw_if_kernel_failed("unsharp mask horizontal gaussian");

        gaussian_vertical_unsharp_kernel<<<grid_size, kImageBlockSize>>>(
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