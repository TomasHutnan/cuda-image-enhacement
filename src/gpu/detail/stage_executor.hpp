#pragma once

#include <functional>
#include <chrono>
#include <string_view>

namespace tgpu::gpu::detail {

/// Framework for standardized stage execution with optional benchmarking.
/// Eliminates repetitive timing/capture boilerplate in pipeline orchestration.
/// 
/// Usage:
///   StageExecutor executor(options.collect_benchmark);
///   executor.execute(
///       options.stage_execution.some_stage,
///       [&]() { run_some_stage(workspace, stage_options); },
///       result.benchmark.some_stage_ms,
///       "some_stage"
///   );
class StageExecutor {
public:
    /// @param collect_benchmark Whether to measure execution time
    explicit StageExecutor(bool collect_benchmark)
        : collect_benchmark_(collect_benchmark) {}

    /// Execute a stage with optional timing and capture preparation.
    /// @param enabled Whether the stage is enabled
    /// @param stage_func Lambda to invoke the stage kernel
    /// @param benchmark_ms Output parameter: stage execution time (ms)
    /// @param stage_name Diagnostic name for logging
    void execute(bool enabled, std::function<void()> stage_func, 
                 double& benchmark_ms, std::string_view stage_name) {
        if (!enabled) {
            benchmark_ms = 0.0;
            return;
        }

        if (collect_benchmark_) {
            benchmark_ms = measure_stage(stage_func);
        } else {
            stage_func();
        }
    }

private:
    /// Measure stage execution time in milliseconds (includes cudaDeviceSynchronize).
    /// @param stage_func Function to measure
    /// @return Execution time in milliseconds
    double measure_stage(std::function<void()> stage_func) {
        cudaDeviceSynchronize();
        
        auto start = std::chrono::high_resolution_clock::now();
        stage_func();
        cudaDeviceSynchronize();
        
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double, std::milli> duration = end - start;
        return duration.count();
    }

    bool collect_benchmark_;
};

}  // namespace tgpu::gpu::detail
