// Richardson-Lucy iterative blind deconvolution stage implementation.

#include "detail/runtime.hpp"
#include "detail/kernel_grid.hpp"
#include "detail/kernel_cache.hpp"

#include <array>
#include <cmath>

namespace tgpu
{
    namespace
    {
        /// Maximum PSF (Point Spread Function) kernel half-width.
        /// Gaussian PSF: radius = ceil(3 * psf_sigma); clamped to [1, kMaxPsfRadius]
        constexpr int kMaxPsfRadius = 31;
        constexpr int kMaxPsfKernelSize = 2 * kMaxPsfRadius + 1;
        __constant__ float kPsf[kMaxPsfKernelSize];

        /// Richardson-Lucy blind deconvolution algorithm.
        /// Iteratively sharpens image assuming Gaussian PSF blur.
        /// Algorithm (per iteration i):
        /// 1. Estimate convolved with PSF: O_i = Estimate * PSF
        /// 2. Ratio: R_i = Observed / (O_i + epsilon)  [prevents division by zero]
        /// 3. Transpose PSF correlation (flip kernel): PSF^T
        /// 4. Estimate update: Estimate_{i+1} = Estimate_i * (R_i * PSF^T)
        ///
        /// More iterations → more deconvolution, risk of ringing artifacts.
        /// Typical: 2-5 iterations; epsilon: 1e-7
        
        // Global PSF cache (device constant memory)
        gpu::detail::KernelCache<float, kMaxPsfKernelSize> g_psf_cache;
        
        // Persistent scratch buffer for RL temp storage
        DeviceFloatBuffer &scratch_buffer_cache()
        {
            static DeviceFloatBuffer scratch;
            return scratch;
        }

        // Helper to compute normalized Gaussian PSF kernel on host
        void compute_gaussian_psf(int radius, float sigma, std::array<float, kMaxPsfKernelSize>& weights)
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

        __global__ void horizontal_convolution_kernel(
            const float *input,
            float *output,
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

            float convolved = 0.0F;
            for (int offset = -radius; offset <= radius; ++offset)
            {
                const int sample_x = clamp_coordinate(x + offset, expanded_width);
                const float weight = kPsf[offset + radius];
                convolved += weight * input[expanded_index(sample_x, y, expanded_width)];
            }

            output[expanded_index(x, y, expanded_width)] = convolved;
        }

        __global__ void vertical_convolution_kernel(
            const float *input,
            float *output,
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

            float convolved = 0.0F;
            for (int offset = -radius; offset <= radius; ++offset)
            {
                const int sample_y = clamp_coordinate(y + offset, expanded_height);
                const float weight = kPsf[offset + radius];
                convolved += weight * input[expanded_index(x, sample_y, expanded_width)];
            }

            output[expanded_index(x, y, expanded_width)] = convolved;
        }

        __global__ void ratio_kernel(
            const float *observed,
            const float *convolved,
            float *ratio,
            std::size_t element_count,
            float epsilon)
        {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                                      static_cast<std::size_t>(threadIdx.x);
            if (index >= element_count)
            {
                return;
            }

            const float denominator = fmaxf(convolved[index], epsilon);
            ratio[index] = observed[index] / denominator;
        }

        __global__ void vertical_update_kernel(
            const float *correction_input,
            float *estimate,
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

            float correction = 0.0F;
            for (int offset = -radius; offset <= radius; ++offset)
            {
                const int sample_y = clamp_coordinate(y + offset, expanded_height);
                const float weight = kPsf[offset + radius];
                correction += weight * correction_input[expanded_index(x, sample_y, expanded_width)];
            }

            const std::size_t index = expanded_index(x, y, expanded_width);
            estimate[index] = fmaxf(estimate[index] * correction, 0.0F);
        }
    } // namespace

    void run_richardson_lucy_stage(const StageWorkspace &workspace, const RichardsonLucyOptions &options)
    {
        if (!options.enabled)
        {
            run_passthrough_stage(workspace, "richardson-lucy disabled");
            return;
        }

        // Validate parameters
        if (options.iterations <= 0)
        {
            run_passthrough_stage(workspace, "RL: iterations must be >= 1");
            return;
        }

        if (!std::isfinite(options.psf_sigma) || options.psf_sigma <= 0.0F)
        {
            run_passthrough_stage(workspace, "RL: psf_sigma must be > 0.0 and finite");
            return;
        }

        if (options.psf_radius < 1)
        {
            run_passthrough_stage(workspace, "RL: psf_radius must be >= 1");
            return;
        }

        if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0F)
        {
            run_passthrough_stage(workspace, "RL: epsilon must be > 0.0 and finite");
            return;
        }

        const int radius = std::clamp(options.psf_radius, 1, kMaxPsfRadius);
        const int kernel_size = 2 * radius + 1;

        // Check/update PSF cache
        if (!g_psf_cache.is_valid(options.psf_sigma, radius))
        {
            std::array<float, kMaxPsfKernelSize> weights{};
            compute_gaussian_psf(radius, options.psf_sigma, weights);

            if (weights[0] <= 0.0F)  // Sanity check
            {
                run_passthrough_stage(workspace, "RL: degenerate PSF kernel");
                return;
            }

            g_psf_cache.update(options.psf_sigma, radius, weights.data(), (void*)kPsf, kernel_size);
        }

        // Allocate/reuse scratch buffer for iteration temporaries
        DeviceFloatBuffer &scratch_buffer = scratch_buffer_cache();
        const std::size_t element_count =
            static_cast<std::size_t>(workspace.expanded_width) * static_cast<std::size_t>(workspace.expanded_height);
        if (scratch_buffer.data == nullptr || scratch_buffer.element_count < element_count)
        {
            scratch_buffer = allocate_float_buffer(element_count);
        }

        // Use standardized grid calculation
        const dim3 grid_2d = gpu::detail::compute_2d_grid(
            workspace.expanded_width,
            workspace.expanded_height,
            gpu::detail::block_sizes::kImage2D_X,
            gpu::detail::block_sizes::kImage2D_Y);

        const int block_count_1d = gpu::detail::compute_1d_grid(
            static_cast<int>(element_count),
            gpu::detail::block_sizes::kLinear);

        // Initialize estimate from observed image
        throw_if_cuda_failed(
            cudaMemcpy(
                workspace.output,
                workspace.input,
                element_count * sizeof(float),
                cudaMemcpyDeviceToDevice),
            "cudaMemcpy richardson-lucy initialize estimate");

        // Iterative deconvolution loop
        for (int iteration = 0; iteration < options.iterations; ++iteration)
        {
            // Forward convolution: Estimate * PSF
            horizontal_convolution_kernel<<<grid_2d, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                           gpu::detail::block_sizes::kImage2D_Y, 1)>>>(
                workspace.output,
                workspace.auxiliary,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy forward horizontal convolution");

            vertical_convolution_kernel<<<grid_2d, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                         gpu::detail::block_sizes::kImage2D_Y, 1)>>>(
                workspace.auxiliary,
                scratch_buffer.data,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy forward vertical convolution");

            // Compute ratio: Observed / (Estimate * PSF + epsilon)
            ratio_kernel<<<block_count_1d, gpu::detail::block_sizes::kLinear>>>(
                workspace.input,
                scratch_buffer.data,
                scratch_buffer.data,
                element_count,
                options.epsilon);
            throw_if_kernel_failed("richardson-lucy ratio");

            // Backward convolution: Ratio * PSF^T
            horizontal_convolution_kernel<<<grid_2d, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                           gpu::detail::block_sizes::kImage2D_Y, 1)>>>(
                scratch_buffer.data,
                workspace.auxiliary,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy backward horizontal convolution");

            // Update estimate
            vertical_update_kernel<<<grid_2d, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                    gpu::detail::block_sizes::kImage2D_Y, 1)>>>(
                workspace.auxiliary,
                workspace.output,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy backward update");
        }
    }
} // namespace tgpu