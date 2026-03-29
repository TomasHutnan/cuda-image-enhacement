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
        __constant__ float kPsf[kMaxPsfKernelSize];

        struct PsfCache
        {
            bool valid = false;
            float sigma = 0.0F;
            int radius = 0;
            std::array<float, kMaxPsfKernelSize> weights{};
        };

        PsfCache &psf_cache()
        {
            static PsfCache cache;
            return cache;
        }

        bool same_sigma(float lhs, float rhs)
        {
            return fabsf(lhs - rhs) <= 1.0e-6F;
        }

        DeviceFloatBuffer &scratch_buffer_cache()
        {
            static DeviceFloatBuffer scratch;
            return scratch;
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

        PsfCache &cache = psf_cache();
        const bool cache_hit =
            cache.valid &&
            cache.radius == radius &&
            same_sigma(cache.sigma, options.psf_sigma);
        if (!cache_hit)
        {
            float sum = 0.0F;
            const float sigma_squared = options.psf_sigma * options.psf_sigma;

            for (int x = 0; x < kernel_size; ++x)
            {
                const int offset_x = x - radius;
                const float distance_squared = static_cast<float>(offset_x * offset_x);
                const float value = expf(-distance_squared / (2.0F * sigma_squared));
                cache.weights[static_cast<std::size_t>(x)] = value;
                sum += value;
            }

            if (sum <= 0.0F)
            {
                run_passthrough_stage(workspace, "richardson-lucy degenerate psf");
                return;
            }

            for (int index = 0; index < kernel_size; ++index)
            {
                cache.weights[static_cast<std::size_t>(index)] /= sum;
            }

            cache.valid = true;
            cache.sigma = options.psf_sigma;
            cache.radius = radius;

            throw_if_cuda_failed(
                cudaMemcpyToSymbol(
                    kPsf,
                    cache.weights.data(),
                    static_cast<std::size_t>(kernel_size) * sizeof(float),
                    0,
                    cudaMemcpyHostToDevice),
                "cudaMemcpyToSymbol richardson-lucy psf");
        }

        DeviceFloatBuffer &scratch_buffer = scratch_buffer_cache();
        const std::size_t element_count =
            static_cast<std::size_t>(workspace.expanded_width) * static_cast<std::size_t>(workspace.expanded_height);
        if (scratch_buffer.data == nullptr || scratch_buffer.element_count < element_count)
        {
            scratch_buffer = allocate_float_buffer(element_count);
        }

        const int block_count = static_cast<int>(
            (element_count + static_cast<std::size_t>(kThreadsPerBlock) - 1) / static_cast<std::size_t>(kThreadsPerBlock));
        const dim3 grid_size{
            static_cast<unsigned int>((workspace.expanded_width + static_cast<int>(kImageBlockSize.x) - 1) /
                                      static_cast<int>(kImageBlockSize.x)),
            static_cast<unsigned int>((workspace.expanded_height + static_cast<int>(kImageBlockSize.y) - 1) /
                                      static_cast<int>(kImageBlockSize.y)),
            1};

        throw_if_cuda_failed(
            cudaMemcpy(
                workspace.output,
                workspace.input,
                element_count * sizeof(float),
                cudaMemcpyDeviceToDevice),
            "cudaMemcpy richardson-lucy initialize estimate");

        for (int iteration = 0; iteration < options.iterations; ++iteration)
        {
            horizontal_convolution_kernel<<<grid_size, kImageBlockSize>>>(
                workspace.output,
                workspace.auxiliary,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy forward horizontal convolution");

            vertical_convolution_kernel<<<grid_size, kImageBlockSize>>>(
                workspace.auxiliary,
                scratch_buffer.data,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy forward vertical convolution");

            ratio_kernel<<<block_count, kThreadsPerBlock>>>(
                workspace.input,
                scratch_buffer.data,
                scratch_buffer.data,
                element_count,
                options.epsilon);
            throw_if_kernel_failed("richardson-lucy ratio");

            horizontal_convolution_kernel<<<grid_size, kImageBlockSize>>>(
                scratch_buffer.data,
                workspace.auxiliary,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy backward horizontal convolution");

            vertical_update_kernel<<<grid_size, kImageBlockSize>>>(
                workspace.auxiliary,
                workspace.output,
                workspace.expanded_width,
                workspace.expanded_height,
                radius);
            throw_if_kernel_failed("richardson-lucy backward update");
        }
    }
} // namespace tgpu