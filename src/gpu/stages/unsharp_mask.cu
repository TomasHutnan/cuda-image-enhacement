// Placeholder for the unsharp mask stage until the CUDA implementation is added.

#include "detail/runtime.hpp"

namespace tgpu
{
    void run_unsharp_mask_stage(const StageWorkspace &workspace)
    {
        run_passthrough_stage(workspace, "unsharp mask placeholder stage");
    }
} // namespace tgpu