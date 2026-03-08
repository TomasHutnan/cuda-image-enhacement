// Placeholder for the histogram stretch stage until the CUDA implementation is added.

#include "detail/runtime.hpp"

namespace tgpu
{
    void run_histogram_stretch_stage(const StageWorkspace &workspace)
    {
        run_passthrough_stage(workspace, "histogram stretch placeholder stage");
    }
} // namespace tgpu