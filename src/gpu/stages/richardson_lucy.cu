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

        struct PsfCache
        {
            bool valid = false;
            float sigma = 0.0F;
            int radius = 0;
            std::array<float, kMaxPsfElements> weights{};
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

        __global__ void psf_convolution_kernel(
            const float *input,
            float *output,
            int expanded_width,
            int expanded_height,
            int radius,
            int kernel_size)
        {
            extern __shared__ float tile[];

            const int local_x = static_cast<int>(threadIdx.x);
            const int local_y = static_cast<int>(threadIdx.y);
            const int base_x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
            const int base_y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y);
            const int x = base_x + local_x;
            const int y = base_y + local_y;

            const int tile_width = static_cast<int>(blockDim.x) + 2 * radius;
            const int tile_height = static_cast<int>(blockDim.y) + 2 * radius;
            const int thread_linear = local_y * static_cast<int>(blockDim.x) + local_x;
            const int thread_count = static_cast<int>(blockDim.x) * static_cast<int>(blockDim.y);
            const int tile_count = tile_width * tile_height;

            for (int index = thread_linear; index < tile_count; index += thread_count)
            {
                const int tile_x = index % tile_width;
                const int tile_y = index / tile_width;
                const int global_x = clamp_coordinate(base_x + tile_x - radius, expanded_width);
                const int global_y = clamp_coordinate(base_y + tile_y - radius, expanded_height);
                tile[index] = input[expanded_index(global_x, global_y, expanded_width)];
            }
            __syncthreads();

            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            float convolved = 0.0F;
            for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y)
            {
                const int sample_y = local_y + kernel_y;
                const int row_offset = sample_y * tile_width;
                const int kernel_offset = kernel_y * kernel_size;
                for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x)
                {
                    const float weight = kPsf[kernel_offset + kernel_x];
                    convolved += weight * tile[row_offset + local_x + kernel_x];
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
            extern __shared__ float tile[];

            const int local_x = static_cast<int>(threadIdx.x);
            const int local_y = static_cast<int>(threadIdx.y);
            const int base_x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x);
            const int base_y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y);
            const int x = base_x + local_x;
            const int y = base_y + local_y;

            const int tile_width = static_cast<int>(blockDim.x) + 2 * radius;
            const int tile_height = static_cast<int>(blockDim.y) + 2 * radius;
            const int thread_linear = local_y * static_cast<int>(blockDim.x) + local_x;
            const int thread_count = static_cast<int>(blockDim.x) * static_cast<int>(blockDim.y);
            const int tile_count = tile_width * tile_height;

            for (int index = thread_linear; index < tile_count; index += thread_count)
            {
                const int tile_x = index % tile_width;
                const int tile_y = index / tile_width;
                const int global_x = clamp_coordinate(base_x + tile_x - radius, expanded_width);
                const int global_y = clamp_coordinate(base_y + tile_y - radius, expanded_height);
                tile[index] = ratio[expanded_index(global_x, global_y, expanded_width)];
            }
            __syncthreads();

            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            float correction = 0.0F;
            for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y)
            {
                const int sample_y = local_y + kernel_y;
                const int row_offset = sample_y * tile_width;
                const int kernel_offset = kernel_y * kernel_size;
                for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x)
                {
                    const float weight = kPsf[kernel_offset + kernel_x];
                    correction += weight * tile[row_offset + local_x + kernel_x];
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

        PsfCache &cache = psf_cache();
        const bool cache_hit =
            cache.valid &&
            cache.radius == radius &&
            same_sigma(cache.sigma, options.psf_sigma);
        if (!cache_hit)
        {
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
                    cache.weights[static_cast<std::size_t>(y * kernel_size + x)] = value;
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
                cache.weights[static_cast<std::size_t>(index)] /= sum;
            }

            cache.valid = true;
            cache.sigma = options.psf_sigma;
            cache.radius = radius;

            throw_if_cuda_failed(
                cudaMemcpyToSymbol(
                    kPsf,
                    cache.weights.data(),
                    static_cast<std::size_t>(kernel_element_count) * sizeof(float),
                    0,
                    cudaMemcpyHostToDevice),
                "cudaMemcpyToSymbol richardson-lucy psf");
        }

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
        const int tile_width = static_cast<int>(kImageBlockSize.x) + 2 * radius;
        const int tile_height = static_cast<int>(kImageBlockSize.y) + 2 * radius;
        const std::size_t shared_tile_bytes = static_cast<std::size_t>(tile_width * tile_height) * sizeof(float);

        throw_if_cuda_failed(
            cudaMemcpy(
                workspace.output,
                workspace.input,
                element_count * sizeof(float),
                cudaMemcpyDeviceToDevice),
            "cudaMemcpy richardson-lucy initialize estimate");

        for (int iteration = 0; iteration < options.iterations; ++iteration)
        {
            psf_convolution_kernel<<<grid_size, kImageBlockSize, shared_tile_bytes>>>(
                workspace.output,
                workspace.auxiliary,
                workspace.expanded_width,
                workspace.expanded_height,
                radius,
                kernel_size);
            throw_if_kernel_failed("richardson-lucy forward convolution");

            ratio_kernel<<<block_count, kThreadsPerBlock>>>(
                workspace.input,
                workspace.auxiliary,
                workspace.auxiliary,
                element_count,
                options.epsilon);
            throw_if_kernel_failed("richardson-lucy ratio");

            backward_update_kernel<<<grid_size, kImageBlockSize, shared_tile_bytes>>>(
                workspace.auxiliary,
                workspace.output,
                workspace.expanded_width,
                workspace.expanded_height,
                radius,
                kernel_size);
            throw_if_kernel_failed("richardson-lucy backward update");
        }
    }
} // namespace tgpu