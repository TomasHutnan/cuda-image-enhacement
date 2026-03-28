#pragma once

#include "cli/cli_types.hpp"

namespace tgpu::cli {

int run_single_file_mode(const CliArguments& arguments);
int run_directory_mode(const CliArguments& arguments);

}  // namespace tgpu::cli
