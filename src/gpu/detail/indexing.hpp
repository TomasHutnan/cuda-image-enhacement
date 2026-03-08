#pragma once

// Shared coordinate and indexing helpers for expanded GPU image buffers.

#include <cuda_runtime.h>

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

    __host__ __device__ __forceinline__ std::size_t expanded_index(int x, int y, int expanded_width)
    {
        return static_cast<std::size_t>(y) * static_cast<std::size_t>(expanded_width) + static_cast<std::size_t>(x);
    }
} // namespace tgpu