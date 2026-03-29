#pragma once

#include <cuda_runtime.h>
#include <cassert>

namespace tgpu::gpu::detail {

/// Compute 2D grid dimensions for image processing kernels.
/// @param width Image width in pixels
/// @param height Image height in pixels
/// @param block_size_x Threads per block in X dimension (default 16)
/// @param block_size_y Threads per block in Y dimension (default 16)
/// @return Grid dimensions (blocks_x, blocks_y, 1)
inline dim3 compute_2d_grid(int width, int height, int block_size_x = 16, int block_size_y = 16) {
    assert(width > 0 && height > 0);
    assert(block_size_x > 0 && block_size_y > 0);
    
    const int blocks_x = (width + block_size_x - 1) / block_size_x;
    const int blocks_y = (height + block_size_y - 1) / block_size_y;
    
    return dim3{static_cast<unsigned int>(blocks_x), 
                static_cast<unsigned int>(blocks_y), 
                1u};
}

/// Compute 1D grid dimensions for linear array processing.
/// @param element_count Total number of elements to process
/// @param threads_per_block Threads per block (default 256)
/// @return Number of blocks needed
inline int compute_1d_grid(int element_count, int threads_per_block = 256) {
    assert(element_count > 0);
    assert(threads_per_block > 0);
    
    return (element_count + threads_per_block - 1) / threads_per_block;
}

/// Compute safe shared memory bin count for histogram operations.
/// @param requested_bins Desired number of histogram bins
/// @param max_shared_memory Maximum shared memory per block (bytes)
/// @param element_size Size of each element (e.g., sizeof(uint))
/// @return Actual bin count (≤ requested_bins, fitting in shared memory)
inline int compute_shared_memory_bins(int requested_bins, int max_shared_memory, int element_size) {
    assert(requested_bins > 0);
    assert(max_shared_memory > 0);
    assert(element_size > 0);
    
    const int max_bins_for_memory = max_shared_memory / element_size;
    return (requested_bins < max_bins_for_memory) ? requested_bins : max_bins_for_memory;
}

/// Recommended block sizes for different kernel types.
namespace block_sizes {
    /// Standard 2D image processing: 16×16 = 256 threads per block.
    static constexpr int kImage2D_X = 16;
    static constexpr int kImage2D_Y = 16;
    
    /// Linear array processing: 256 threads per block.
    static constexpr int kLinear = 256;
}

}  // namespace tgpu::gpu::detail
