#pragma once

// Private CUDA pipeline data structures shared across the GPU implementation files.

#include "tgpu/pipeline.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace tgpu
{
    struct DeviceFloatBuffer
    {
        float *data = nullptr;
        std::size_t element_count = 0;

        DeviceFloatBuffer() = default;
        DeviceFloatBuffer(const DeviceFloatBuffer &) = delete;
        DeviceFloatBuffer &operator=(const DeviceFloatBuffer &) = delete;

        DeviceFloatBuffer(DeviceFloatBuffer &&other) noexcept : data(other.data), element_count(other.element_count)
        {
            other.data = nullptr;
            other.element_count = 0;
        }

        DeviceFloatBuffer &operator=(DeviceFloatBuffer &&other) noexcept
        {
            if (this == &other)
            {
                return *this;
            }

            if (data != nullptr)
            {
                cudaFree(data);
            }

            data = other.data;
            element_count = other.element_count;
            other.data = nullptr;
            other.element_count = 0;
            return *this;
        }

        ~DeviceFloatBuffer()
        {
            if (data != nullptr)
            {
                cudaFree(data);
            }
        }
    };

    struct DeviceByteBuffer
    {
        std::uint8_t *data = nullptr;
        std::size_t byte_count = 0;

        DeviceByteBuffer() = default;
        DeviceByteBuffer(const DeviceByteBuffer &) = delete;
        DeviceByteBuffer &operator=(const DeviceByteBuffer &) = delete;

        DeviceByteBuffer(DeviceByteBuffer &&other) noexcept : data(other.data), byte_count(other.byte_count)
        {
            other.data = nullptr;
            other.byte_count = 0;
        }

        ~DeviceByteBuffer()
        {
            if (data != nullptr)
            {
                cudaFree(data);
            }
        }
    };

    struct DevicePipeline
    {
        int width = 0;
        int height = 0;
        int border = 0;
        int expanded_width = 0;
        int expanded_height = 0;
        DeviceFloatBuffer buffers[3];
        float *current = nullptr;
        float *next = nullptr;
        float *scratch = nullptr;
    };

    struct StageWorkspace
    {
        const float *input = nullptr;
        float *output = nullptr;
        float *auxiliary = nullptr;
        int width = 0;
        int height = 0;
        int border = 0;
        int expanded_width = 0;
        int expanded_height = 0;
    };

    constexpr int kThreadsPerBlock = 256;
    constexpr dim3 kImageBlockSize{16, 16, 1};

} // namespace tgpu