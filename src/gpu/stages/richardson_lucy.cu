// Placeholder for the Richardson-Lucy stage until the CUDA implementation is added.

#include "detail/runtime.hpp"

#include <array>
#include <cmath>

namespace tgpu
{
    namespace
    {
        constexpr int kMaxPsfRadius = 31;
        constexpr int kMaxPsfKernelSize = 2 * kMaxPsfRadius + 1;
        constexpr int kMaxPsfElements = kMaxPsfKernelSize * kMaxPsfKernelSize;

        __constant__ float kPsf[kMaxPsfElements];

        __global__ void copy_kernel(const float *input, float *output, std::size_t count)
        {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                                      static_cast<std::size_t>(threadIdx.x);
            if (index < count)
            {
                output[index] = input[index];
            }
        }

        __global__ void psf_convolution_kernel(
            const float *input,
            float *output,
            int expanded_width,
            int expanded_height,
            int radius,
            int kernel_size)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            float convolved = 0.0F;
            for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y)
            {
                const int sample_y = clamp_coordinate(y + kernel_y - radius, expanded_height);
                for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x)
                {
                    const int sample_x = clamp_coordinate(x + kernel_x - radius, expanded_width);
                    const float weight = kPsf[kernel_y * kernel_size + kernel_x];
                    convolved += weight * input[expanded_index(sample_x, sample_y, expanded_width)];
                }
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

        __global__ void backward_update_kernel(
            const float *ratio,
            float *estimate,
            int expanded_width,
            int expanded_height,
            int radius,
            int kernel_size)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            float correction = 0.0F;
            for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y)
            {
                const int sample_y = clamp_coordinate(y + kernel_y - radius, expanded_height);
                for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x)
                {
                    const int sample_x = clamp_coordinate(x + kernel_x - radius, expanded_width);
                    const float weight = kPsf[kernel_y * kernel_size + kernel_x];
                    correction += weight * ratio[expanded_index(sample_x, sample_y, expanded_width)];
                }
            }

            const std::size_t index = expanded_index(x, y, expanded_width);
            estimate[index] = fmaxf(estimate[index] * correction, 0.0F);
        }
    } // namespace

    void run_richardson_lucy_stage(const StageWorkspace &workspace, const RichardsonLucyOptions &options)
    {
        if (options.iterations <= 0)
        {
            run_passthrough_stage(workspace, "richardson-lucy disabled iterations");
            return;
        }

        if (!std::isfinite(options.psf_sigma) || options.psf_sigma <= 0.0F)
        {
            run_passthrough_stage(workspace, "richardson-lucy invalid sigma");
            return;
        }

        if (!std::isfinite(options.epsilon) || options.epsilon <= 0.0F)
        {
            run_passthrough_stage(workspace, "richardson-lucy invalid epsilon");
            return;
        }

        const int radius = std::clamp(options.psf_radius, 1, kMaxPsfRadius);
        const int kernel_size = 2 * radius + 1;
        const int kernel_element_count = kernel_size * kernel_size;

        std::array<float, kMaxPsfElements> host_psf{};
        float sum = 0.0F;
        const float sigma_squared = options.psf_sigma * options.psf_sigma;

        for (int y = 0; y < kernel_size; ++y)
        {
            const int offset_y = y - radius;
            for (int x = 0; x < kernel_size; ++x)
            {
                const int offset_x = x - radius;
                const float distance_squared = static_cast<float>(offset_x * offset_x + offset_y * offset_y);
                const float value = expf(-distance_squared / (2.0F * sigma_squared));
                host_psf[static_cast<std::size_t>(y * kernel_size + x)] = value;
                sum += value;
            }
        }

        if (sum <= 0.0F)
        {
            run_passthrough_stage(workspace, "richardson-lucy degenerate psf");
            return;
        }

        for (int index = 0; index < kernel_element_count; ++index)
        {
            host_psf[static_cast<std::size_t>(index)] /= sum;
        }

        throw_if_cuda_failed(
            cudaMemcpyToSymbol(
                kPsf,
                host_psf.data(),
                static_cast<std::size_t>(kernel_element_count) * sizeof(float),
                0,
                cudaMemcpyHostToDevice),
            "cudaMemcpyToSymbol richardson-lucy psf");

        const std::size_t element_count =
            static_cast<std::size_t>(workspace.expanded_width) * static_cast<std::size_t>(workspace.expanded_height);
        const int block_count = static_cast<int>(
            (element_count + static_cast<std::size_t>(kThreadsPerBlock) - 1) / static_cast<std::size_t>(kThreadsPerBlock));
        const dim3 grid_size{
            static_cast<unsigned int>((workspace.expanded_width + static_cast<int>(kImageBlockSize.x) - 1) /
                                      static_cast<int>(kImageBlockSize.x)),
            static_cast<unsigned int>((workspace.expanded_height + static_cast<int>(kImageBlockSize.y) - 1) /
                                      static_cast<int>(kImageBlockSize.y)),
            1};

        copy_kernel<<<block_count, kThreadsPerBlock>>>(workspace.input, workspace.auxiliary, element_count);
        throw_if_kernel_failed("richardson-lucy initialize estimate");

        for (int iteration = 0; iteration < options.iterations; ++iteration)
        {
            psf_convolution_kernel<<<grid_size, kImageBlockSize>>>(
                workspace.auxiliary,
                workspace.output,
                workspace.expanded_width,
                workspace.expanded_height,
                radius,
                kernel_size);
            throw_if_kernel_failed("richardson-lucy forward convolution");

            ratio_kernel<<<block_count, kThreadsPerBlock>>>(
                workspace.input,
                workspace.output,
                workspace.output,
                element_count,
                options.epsilon);
            throw_if_kernel_failed("richardson-lucy ratio");

            backward_update_kernel<<<grid_size, kImageBlockSize>>>(
                workspace.output,
                workspace.auxiliary,
                workspace.expanded_width,
                workspace.expanded_height,
                radius,
                kernel_size);
            throw_if_kernel_failed("richardson-lucy backward update");
        }

        copy_kernel<<<block_count, kThreadsPerBlock>>>(workspace.auxiliary, workspace.output, element_count);
        throw_if_kernel_failed("richardson-lucy finalize output");
    }
} // namespace tgpu