#pragma once

#include <string>
#include <vector>

#include "cli/cli_types.hpp"

namespace tgpu::cli {

std::string format_stage_file_name(const tgpu::PipelineStage& stage);
void print_benchmark_report(const tgpu::PipelineRunResult& result, const tgpu::PipelineRunOptions& options);
void print_mean_benchmark_report(const std::vector<ProcessingResult>& results, const tgpu::PipelineRunOptions& options);

}  // namespace tgpu::cli
