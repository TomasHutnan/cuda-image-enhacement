#pragma once

// Shared coordinate, indexing, and grid helpers for GPU kernels.

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>

namespace tgpu
{
    __host__ __device__ __forceinline__ int clamp_coordinate(int value, int limit)
    {
        if (value < 0)
        {
            return 0;
        }
        if (value >= limit)
        {
            return limit - 1;
        }
        return value;
    }

    __host__ __device__ __forceinline__ int reflect_coordinate(int value, int limit)
    {
        // Symmetric reflection (reflects around boundaries at 0 and limit-1)
        // Maps: ..., -2→2, -1→1, 0→0, 1→1, 2→2, 3→2, 4→1, 5→0, ...
        if (limit <= 0)
        {
            return 0;
        }
        if (limit == 1)
        {
            return 0;
        }
        const int period = 2 * (limit - 1);
        const int modulo = abs(value) % period;
        return modulo < limit ? modulo : period - modulo;
    }

    __host__ __device__ __forceinline__ std::size_t expanded_index(int x, int y, int expanded_width)
    {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(expanded_width) + static_cast<std::size_t>(x);
    }
} // namespace tgpu

namespace tgpu::gpu::detail
{
    inline dim3 compute_2d_grid(int width, int height, int block_size_x = 16, int block_size_y = 16)
    {
        assert(width > 0 && height > 0);
        assert(block_size_x > 0 && block_size_y > 0);

        const int blocks_x = (width + block_size_x - 1) / block_size_x;
        const int blocks_y = (height + block_size_y - 1) / block_size_y;

        return dim3{static_cast<unsigned int>(blocks_x),
                    static_cast<unsigned int>(blocks_y),
                    1u};
    }

    inline int compute_1d_grid(int element_count, int threads_per_block = 256)
    {
        assert(element_count > 0);
        assert(threads_per_block > 0);

        return (element_count + threads_per_block - 1) / threads_per_block;
    }

    inline int compute_shared_memory_bins(int requested_bins, int max_shared_memory, int element_size)
    {
        assert(requested_bins > 0);
        assert(max_shared_memory > 0);
        assert(element_size > 0);

        const int max_bins_for_memory = max_shared_memory / element_size;
        return (requested_bins < max_bins_for_memory) ? requested_bins : max_bins_for_memory;
    }

    namespace block_sizes
    {
        static constexpr int kImage2D_X = 16;
        static constexpr int kImage2D_Y = 16;
        static constexpr int kLinear = 256;
    }

} // namespace tgpu::gpu::detail