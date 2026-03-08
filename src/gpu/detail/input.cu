// Uploads host images and prepares normalized, border-expanded GPU inputs.

#include "detail/runtime.hpp"

namespace tgpu
{
    namespace
    {
        __global__ void normalize_u8_to_expanded_kernel(
            const std::uint8_t *input,
            int width,
            int height,
            int stride_bytes,
            int border,
            int expanded_width,
            int expanded_height,
            float *output)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            const int src_x = min(max(x - border, 0), width - 1);
            const int src_y = min(max(y - border, 0), height - 1);
            const std::size_t src_index = static_cast<std::size_t>(src_y) * static_cast<std::size_t>(stride_bytes) +
                                          static_cast<std::size_t>(src_x);
            const std::size_t dst_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(expanded_width) +
                                          static_cast<std::size_t>(x);
            output[dst_index] = static_cast<float>(input[src_index]) / 255.0F;
        }

        __global__ void normalize_u16_to_expanded_kernel(
            const std::uint16_t *input,
            int width,
            int height,
            int stride_pixels,
            int border,
            int expanded_width,
            int expanded_height,
            float *output)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            const int src_x = min(max(x - border, 0), width - 1);
            const int src_y = min(max(y - border, 0), height - 1);
            const std::size_t src_index = static_cast<std::size_t>(src_y) * static_cast<std::size_t>(stride_pixels) +
                                          static_cast<std::size_t>(src_x);
            const std::size_t dst_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(expanded_width) +
                                          static_cast<std::size_t>(x);
            output[dst_index] = static_cast<float>(input[src_index]) / 65535.0F;
        }

        __global__ void pad_normalized_f32_kernel(
            const float *input,
            int width,
            int height,
            int stride,
            int border,
            int expanded_width,
            int expanded_height,
            float *output)
        {
            const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
            const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
            if (x >= expanded_width || y >= expanded_height)
            {
                return;
            }

            const int src_x = min(max(x - border, 0), width - 1);
            const int src_y = min(max(y - border, 0), height - 1);
            const std::size_t src_index = static_cast<std::size_t>(src_y) * static_cast<std::size_t>(stride) +
                                          static_cast<std::size_t>(src_x);
            const std::size_t dst_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(expanded_width) +
                                          static_cast<std::size_t>(x);
            output[dst_index] = input[src_index];
        }
    } // namespace

    void initialize_pipeline_from_raw(const ImageGray &input, DevicePipeline &pipeline)
    {
        const DeviceByteBuffer source = upload_raw_image(input);
        const dim3 grid_size{
            static_cast<unsigned int>((pipeline.expanded_width + static_cast<int>(kImageBlockSize.x) - 1) /
                                      static_cast<int>(kImageBlockSize.x)),
            static_cast<unsigned int>((pipeline.expanded_height + static_cast<int>(kImageBlockSize.y) - 1) /
                                      static_cast<int>(kImageBlockSize.y)),
            1};

        if (input.bit_depth == BitDepth::u8)
        {
            normalize_u8_to_expanded_kernel<<<grid_size, kImageBlockSize>>>(
                source.data,
                input.width,
                input.height,
                input.stride_bytes,
                pipeline.border,
                pipeline.expanded_width,
                pipeline.expanded_height,
                pipeline.current);
        }
        else
        {
            normalize_u16_to_expanded_kernel<<<grid_size, kImageBlockSize>>>(
                reinterpret_cast<const std::uint16_t *>(source.data),
                input.width,
                input.height,
                input.stride_bytes / static_cast<int>(sizeof(std::uint16_t)),
                pipeline.border,
                pipeline.expanded_width,
                pipeline.expanded_height,
                pipeline.current);
        }

        throw_if_kernel_failed("normalize and pad raw input");
    }

    void initialize_pipeline_from_normalized(const ImageF32 &input, DevicePipeline &pipeline)
    {
        const DeviceFloatBuffer source = upload_normalized_image(input);
        const dim3 grid_size{
            static_cast<unsigned int>((pipeline.expanded_width + static_cast<int>(kImageBlockSize.x) - 1) /
                                      static_cast<int>(kImageBlockSize.x)),
            static_cast<unsigned int>((pipeline.expanded_height + static_cast<int>(kImageBlockSize.y) - 1) /
                                      static_cast<int>(kImageBlockSize.y)),
            1};

        pad_normalized_f32_kernel<<<grid_size, kImageBlockSize>>>(
            source.data,
            input.width,
            input.height,
            input.stride,
            pipeline.border,
            pipeline.expanded_width,
            pipeline.expanded_height,
            pipeline.current);
        throw_if_kernel_failed("pad normalized host input");
    }
} // namespace tgpu