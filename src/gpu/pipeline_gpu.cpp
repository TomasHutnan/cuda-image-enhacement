// Public pipeline entrypoints that forward into the CUDA implementation.

#include "tgpu/pipeline.hpp"
#include "detail/pipeline_api.hpp"

#include <cuda_runtime.h>

namespace tgpu
{
    namespace
    {
        void free_device_image(DeviceImageF32 &image) noexcept
        {
            if (image.data != nullptr)
            {
                cudaFree(image.data);
                image.data = nullptr;
            }
            image.width = 0;
            image.height = 0;
            image.stride = 0;
        }
    }

    DeviceImageF32::DeviceImageF32(DeviceImageF32 &&other) noexcept
        : data(other.data), width(other.width), height(other.height), stride(other.stride)
    {
        other.data = nullptr;
        other.width = 0;
        other.height = 0;
        other.stride = 0;
    }

    DeviceImageF32 &DeviceImageF32::operator=(DeviceImageF32 &&other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        free_device_image(*this);
        data = other.data;
        width = other.width;
        height = other.height;
        stride = other.stride;

        other.data = nullptr;
        other.width = 0;
        other.height = 0;
        other.stride = 0;
        return *this;
    }

    DeviceImageF32::~DeviceImageF32()
    {
        free_device_image(*this);
    }

    PipelineRunResult run_pipeline(const ImageGray &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda(input, options);
    }

    PipelineRunResult run_pipeline(const ImageF32 &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda(input, options);
    }

    PipelineRunDeviceResult run_pipeline_device(const ImageGray &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda_device(input, options);
    }

    PipelineRunDeviceResult run_pipeline_device(const ImageF32 &input, const PipelineRunOptions &options)
    {
        return run_pipeline_cuda_device(input, options);
    }

    void begin_pipeline_batch(int width, int height)
    {
        begin_pipeline_batch_cuda(width, height);
    }

    void end_pipeline_batch()
    {
        end_pipeline_batch_cuda();
    }
} // namespace tgpu
