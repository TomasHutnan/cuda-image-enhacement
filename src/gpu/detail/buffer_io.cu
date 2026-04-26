// Owns device buffer allocation, host input initialization, and output capture helpers.

#include "detail/runtime.hpp"
#include "detail/pipeline_api.hpp"

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

    void initialize_pipeline_from_raw(const ImageGray &input, DevicePipeline &pipeline)
    {
        const DeviceByteBuffer source = upload_raw_image(input);
        const dim3 grid_size = gpu::detail::compute_2d_grid(
            pipeline.expanded_width,
            pipeline.expanded_height,
            static_cast<int>(kImageBlockSize.x),
            static_cast<int>(kImageBlockSize.y));

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
        const dim3 grid_size = gpu::detail::compute_2d_grid(
            pipeline.expanded_width,
            pipeline.expanded_height,
            static_cast<int>(kImageBlockSize.x),
            static_cast<int>(kImageBlockSize.y));

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

    ImageF32 download_visible_region(const DevicePipeline &pipeline, const float *source)
    {
        ImageF32 image;
        image.width = pipeline.width;
        image.height = pipeline.height;
        image.stride = pipeline.width;
        image.data.resize(static_cast<std::size_t>(image.width) * static_cast<std::size_t>(image.height));

        const float *visible_source = source +
                                      static_cast<std::size_t>(pipeline.border) * static_cast<std::size_t>(pipeline.expanded_width) +
                                      static_cast<std::size_t>(pipeline.border);
        throw_if_cuda_failed(
            cudaMemcpy2D(
                image.data.data(),
                static_cast<std::size_t>(image.stride) * sizeof(float),
                visible_source,
                static_cast<std::size_t>(pipeline.expanded_width) * sizeof(float),
                static_cast<std::size_t>(image.width) * sizeof(float),
                static_cast<std::size_t>(image.height),
                cudaMemcpyDeviceToHost),
            "cudaMemcpy2D download visible image");
        return image;
    }

    DeviceImageF32 download_visible_region_to_device(const DevicePipeline &pipeline, const float *source)
    {
        DeviceImageF32 image;
        image.width = pipeline.width;
        image.height = pipeline.height;
        image.stride = pipeline.width;

        const std::size_t row_bytes = static_cast<std::size_t>(image.width) * sizeof(float);
        const std::size_t total_bytes = row_bytes * static_cast<std::size_t>(image.height);
        throw_if_cuda_failed(cudaMalloc(&image.data, total_bytes), "cudaMalloc visible device image");

        const float *visible_source = source +
                                      static_cast<std::size_t>(pipeline.border) * static_cast<std::size_t>(pipeline.expanded_width) +
                                      static_cast<std::size_t>(pipeline.border);
        throw_if_cuda_failed(
            cudaMemcpy2D(
                image.data,
                row_bytes,
                visible_source,
                static_cast<std::size_t>(pipeline.expanded_width) * sizeof(float),
                row_bytes,
                static_cast<std::size_t>(image.height),
                cudaMemcpyDeviceToDevice),
            "cudaMemcpy2D visible device image copy");
        return image;
    }

    void capture_stage_if_requested(
        PipelineRunResult &result,
        const PipelineRunOptions &options,
        const DevicePipeline &pipeline,
        std::uint32_t prefix,
        std::string_view name,
        const float *source)
    {
        if (!options.capture_intermediate_stages)
        {
            return;
        }

        result.stages.push_back(PipelineStage{prefix, std::string{name}, download_visible_region(pipeline, source)});
    }

} // namespace tgpu