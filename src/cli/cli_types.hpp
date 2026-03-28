#pragma once

#include <filesystem>

#include "tgpu/pipeline.hpp"

namespace tgpu::cli {

struct CliArguments {
    std::filesystem::path input_path;
    std::filesystem::path output_path;
    tgpu::BitDepth output_depth = tgpu::BitDepth::u8;
    std::filesystem::path stages_output_dir;
    bool dump_stages = false;
    tgpu::PipelineRunOptions pipeline_options;
};

struct ProcessingResult {
    bool success = false;
    std::filesystem::path input_file;
    tgpu::PipelineRunResult pipeline_result;
};

}  // namespace tgpu::cli
