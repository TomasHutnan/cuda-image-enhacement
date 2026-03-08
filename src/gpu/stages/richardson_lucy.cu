// Placeholder for the Richardson-Lucy stage until the CUDA implementation is added.

#include "detail/runtime.hpp"

namespace tgpu
{
    void run_richardson_lucy_stage(const StageWorkspace &workspace)
    {
        run_passthrough_stage(workspace, "richardson-lucy placeholder stage");
    }
} // namespace tgpu