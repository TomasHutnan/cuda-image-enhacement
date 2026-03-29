// Histogram stretch stage based on percentile clipping in normalized [0, 1] space.

#include "detail/runtime.hpp"
#include "detail/compute.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace tgpu
{
    namespace
    {
        __global__ void histogram_visible_region_kernel(
            const float *input,
            int width,
            int height,
            int border,
            int expanded_width,
            unsigned int *histogram,
            int bins)
        {
            extern __shared__ unsigned int local_histogram[];

            const int thread_linear_index =
                static_cast<int>(threadIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int threads_per_block = static_cast<int>(blockDim.x) * static_cast<int>(blockDim.y);

            for (int bin = thread_linear_index; bin < bins; bin += threads_per_block)
            {
                local_histogram[bin] = 0U;
            }
            __syncthreads();

            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x < width && y < height)
            {
                const int expanded_x = x + border;
                const int expanded_y = y + border;
                const float value = input[expanded_index(expanded_x, expanded_y, expanded_width)];
                const float clamped = fminf(fmaxf(value, 0.0F), 1.0F);
                int bin = static_cast<int>(clamped * static_cast<float>(bins - 1));
                bin = min(max(bin, 0), bins - 1);
                atomicAdd(&local_histogram[bin], 1U);
            }
            __syncthreads();

            for (int bin = thread_linear_index; bin < bins; bin += threads_per_block)
            {
                const unsigned int count = local_histogram[bin];
                if (count > 0U)
                {
                    atomicAdd(&histogram[bin], count);
                }
            }
        }

        __global__ void apply_histogram_stretch_kernel(
            const float *input,
            float *output,
            std::size_t element_count,
            float low_value,
            float high_value)
        {
            const std::size_t index = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
                                      static_cast<std::size_t>(threadIdx.x);
            if (index >= element_count)
            {
                return;
            }

            const float normalized = (input[index] - low_value) / (high_value - low_value);
            output[index] = fminf(fmaxf(normalized, 0.0F), 1.0F);
        }

        float bin_to_value(int bin, int bins)
        {
            if (bins <= 1)
            {
                return 0.0F;
            }
            return static_cast<float>(bin) / static_cast<float>(bins - 1);
        }

    } // namespace

    void run_histogram_stretch_stage(const StageWorkspace &workspace, const HistogramStretchOptions &options)
    {
        if (!options.enabled)
        {
            run_passthrough_stage(workspace, "histogram stretch disabled");
            return;
        }

        // Validate parameters
        if (options.saturation_percent < 0.0F || options.saturation_percent >= 50.0F)
        {
            run_passthrough_stage(workspace, "Histogram: saturation_percent must be in [0, 50)");
            return;
        }

        if (options.bin_count < 64)
        {
            run_passthrough_stage(workspace, "Histogram: bin_count must be >= 64");
            return;
        }

        const float saturation_percent = std::min(std::max(options.saturation_percent, 0.0F), 49.9F);
        const std::size_t expanded_element_count =
            static_cast<std::size_t>(workspace.expanded_width) * static_cast<std::size_t>(workspace.expanded_height);

        // Query device shared memory limit and compute safe bin count
        int max_shared_memory_bytes = 0;
        throw_if_cuda_failed(
            cudaDeviceGetAttribute(&max_shared_memory_bytes, cudaDevAttrMaxSharedMemoryPerBlock, 0),
            "cudaDeviceGetAttribute max shared memory");
        
        // Constrain bins by both shared memory and workspace buffer limits
        const int max_bins_by_shared = std::max(max_shared_memory_bytes / static_cast<int>(sizeof(unsigned int)), 1);
        const int max_bins_by_workspace =
            static_cast<int>(std::min(expanded_element_count, static_cast<std::size_t>(std::numeric_limits<int>::max())));
        const int max_bins = std::max(std::min(max_bins_by_shared, max_bins_by_workspace), 1);
        const int bins = std::min(std::max(options.bin_count, 64), max_bins);

        auto *device_histogram = reinterpret_cast<unsigned int *>(workspace.auxiliary);
        throw_if_cuda_failed(
            cudaMemset(device_histogram, 0, static_cast<std::size_t>(bins) * sizeof(unsigned int)),
            "cudaMemset histogram bins");

        // Use standardized grid calculation for visible region histogram building
        const dim3 grid_size = gpu::detail::compute_2d_grid(
            workspace.width,
            workspace.height);

        histogram_visible_region_kernel<<<grid_size, dim3(gpu::detail::block_sizes::kImage2D_X, 
                                                           gpu::detail::block_sizes::kImage2D_Y, 1),
                                           static_cast<std::size_t>(bins) * sizeof(unsigned int)>>>(
            workspace.input,
            workspace.width,
            workspace.height,
            workspace.border,
            workspace.expanded_width,
            device_histogram,
            bins);
        throw_if_kernel_failed("histogram stretch histogram build");

        std::vector<unsigned int> histogram(static_cast<std::size_t>(bins));
        throw_if_cuda_failed(
            cudaMemcpy(
                histogram.data(),
                device_histogram,
                static_cast<std::size_t>(bins) * sizeof(unsigned int),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy histogram bins");

        const std::uint64_t visible_count =
            static_cast<std::uint64_t>(workspace.width) * static_cast<std::uint64_t>(workspace.height);
        const std::uint64_t last_rank = visible_count > 0 ? visible_count - 1 : 0;
        const double low_fraction = static_cast<double>(saturation_percent) / 100.0;
        const double high_fraction = 1.0 - low_fraction;

        const std::uint64_t low_rank = static_cast<std::uint64_t>(std::floor(low_fraction * static_cast<double>(last_rank)));
        const std::uint64_t high_rank = static_cast<std::uint64_t>(std::floor(high_fraction * static_cast<double>(last_rank)));

        std::uint64_t cumulative = 0;
        int low_bin = static_cast<int>(histogram.size()) - 1;
        int high_bin = static_cast<int>(histogram.size()) - 1;
        bool low_found = false;
        for (int bin = 0; bin < static_cast<int>(histogram.size()); ++bin)
        {
            cumulative += static_cast<std::uint64_t>(histogram[bin]);
            if (!low_found && cumulative > low_rank)
            {
                low_bin = bin;
                low_found = true;
            }
            if (cumulative > high_rank)
            {
                high_bin = bin;
                break;
            }
        }

        const float low_value = bin_to_value(low_bin, bins);
        const float high_value = bin_to_value(high_bin, bins);

        if (high_value - low_value <= 1.0e-6F)
        {
            run_passthrough_stage(workspace, "histogram stretch flat image");
            return;
        }

        // Use standardized 1D grid for pixel-wise histogram stretch
        const std::size_t element_count = expanded_element_count;
        const int block_count = gpu::detail::compute_1d_grid(
            static_cast<int>(element_count));

        apply_histogram_stretch_kernel<<<block_count, gpu::detail::block_sizes::kLinear>>>(
            workspace.input,
            workspace.output,
            element_count,
            low_value,
            high_value);
        throw_if_kernel_failed("histogram stretch apply");
    }
} // namespace tgpu