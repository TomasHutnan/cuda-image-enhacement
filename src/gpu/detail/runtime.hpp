#pragma once

// Private CUDA runtime helpers used across GPU implementation files.

#include "detail/compute.hpp"
#include "detail/buffers.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace tgpu
{
    inline void throw_if_cuda_failed(cudaError_t status, const char *operation)
    {
        if (status == cudaSuccess)
        {
            return;
        }

        throw std::runtime_error(std::string{operation} + ": " + cudaGetErrorString(status));
    }

    inline bool &strict_kernel_sync_checks_flag()
    {
        static thread_local bool enabled = false;
        return enabled;
    }

    inline void set_strict_kernel_sync_checks(bool enabled)
    {
        strict_kernel_sync_checks_flag() = enabled;
    }

    inline bool strict_kernel_sync_checks_enabled()
    {
        return strict_kernel_sync_checks_flag();
    }

    inline void throw_if_kernel_failed(const char *operation)
    {
        throw_if_cuda_failed(cudaPeekAtLastError(), operation);
        if (strict_kernel_sync_checks_enabled())
        {
            throw_if_cuda_failed(cudaDeviceSynchronize(), operation);
        }
    }

    inline int compute_border(int width, int height)
    {
        return std::max(width / 5, height / 5);
    }

    inline StageWorkspace stage_workspace(const DevicePipeline &pipeline)
    {
        return StageWorkspace{
            .input = pipeline.current,
            .output = pipeline.next,
            .auxiliary = pipeline.scratch,
            .width = pipeline.width,
            .height = pipeline.height,
            .border = pipeline.border,
            .expanded_width = pipeline.expanded_width,
            .expanded_height = pipeline.expanded_height,
        };
    }

    inline void commit_stage_output(DevicePipeline &pipeline)
    {
        float *previous_input = pipeline.current;
        pipeline.current = pipeline.next;
        pipeline.next = previous_input;
    }

    // Passthrough is used by stage files for validation and fail-safe behavior.
    void run_passthrough_stage(const StageWorkspace &workspace, const char *operation);

} // namespace tgpu