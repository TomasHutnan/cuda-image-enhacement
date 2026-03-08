// Owns device buffer allocation and host-to-device upload helpers.

#include "detail/runtime.hpp"

namespace tgpu
{
    DeviceFloatBuffer allocate_float_buffer(std::size_t element_count)
    {
        DeviceFloatBuffer buffer;
        buffer.element_count = element_count;
        throw_if_cuda_failed(cudaMalloc(&buffer.data, element_count * sizeof(float)), "cudaMalloc float buffer");
        return buffer;
    }

    DeviceByteBuffer upload_raw_image(const ImageGray &input)
    {
        DeviceByteBuffer buffer;
        buffer.byte_count = input.data.size();
        throw_if_cuda_failed(cudaMalloc(&buffer.data, buffer.byte_count), "cudaMalloc raw image buffer");
        throw_if_cuda_failed(cudaMemcpy(buffer.data, input.data.data(), buffer.byte_count, cudaMemcpyHostToDevice),
                             "cudaMemcpy raw image upload");
        return buffer;
    }

    DeviceFloatBuffer upload_normalized_image(const ImageF32 &input)
    {
        DeviceFloatBuffer buffer = allocate_float_buffer(input.data.size());
        throw_if_cuda_failed(
            cudaMemcpy(
                buffer.data,
                input.data.data(),
                input.data.size() * sizeof(float),
                cudaMemcpyHostToDevice),
            "cudaMemcpy normalized image upload");
        return buffer;
    }

    DevicePipeline create_pipeline(int width, int height)
    {
        DevicePipeline pipeline;
        pipeline.width = width;
        pipeline.height = height;
        pipeline.border = compute_border(width, height);
        pipeline.expanded_width = width + 2 * pipeline.border;
        pipeline.expanded_height = height + 2 * pipeline.border;

        const std::size_t expanded_element_count =
            static_cast<std::size_t>(pipeline.expanded_width) * static_cast<std::size_t>(pipeline.expanded_height);
        pipeline.buffers[0] = allocate_float_buffer(expanded_element_count);
        pipeline.buffers[1] = allocate_float_buffer(expanded_element_count);
        pipeline.buffers[2] = allocate_float_buffer(expanded_element_count);
        pipeline.current = pipeline.buffers[0].data;
        pipeline.next = pipeline.buffers[1].data;
        pipeline.scratch = pipeline.buffers[2].data;
        return pipeline;
    }
} // namespace tgpu