// Contains the non-local means stage implementation and stage-local kernels.

#include "detail/runtime.hpp"

namespace tgpu
{
    void run_non_local_means_stage(const StageWorkspace &workspace)
    {
        run_passthrough_stage(workspace, "richardson-lucy placeholder stage");
    }
} // namespace tgpu