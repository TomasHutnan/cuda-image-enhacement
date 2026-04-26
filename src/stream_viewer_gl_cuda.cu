#include "stream_viewer_gl.hpp"

#include <cuda_runtime.h>

namespace tgpu::stream {

namespace {

__global__ void f32_to_rgba_kernel(const float* src,
                                   int width,
                                   int height,
                                   int src_stride,
                                   unsigned char* dst_rgba)
{
    const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
    if (x >= width || y >= height)
    {
        return;
    }

    const std::size_t src_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride) +
                                  static_cast<std::size_t>(x);
    const std::size_t dst_index = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                   static_cast<std::size_t>(x)) * 4U;

    const float value = src[src_index];
    const float clamped = fminf(fmaxf(value, 0.0F), 1.0F);
    const unsigned char gray = static_cast<unsigned char>(clamped * 255.0F);
    dst_rgba[dst_index + 0] = gray;
    dst_rgba[dst_index + 1] = gray;
    dst_rgba[dst_index + 2] = gray;
    dst_rgba[dst_index + 3] = 255;
}

}  // namespace

void launch_f32_to_rgba_kernel(const float* src,
                               int width,
                               int height,
                               int src_stride,
                               unsigned char* dst_rgba)
{
    const dim3 block_size{16, 16, 1};
    const dim3 grid_size{
        static_cast<unsigned int>((width + static_cast<int>(block_size.x) - 1) / static_cast<int>(block_size.x)),
        static_cast<unsigned int>((height + static_cast<int>(block_size.y) - 1) / static_cast<int>(block_size.y)),
        1,
    };

    f32_to_rgba_kernel<<<grid_size, block_size>>>(src, width, height, src_stride, dst_rgba);
}

}  // namespace tgpu::stream
