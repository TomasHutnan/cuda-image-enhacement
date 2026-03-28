#pragma once

#include <iosfwd>

#include "cli/cli_types.hpp"

namespace tgpu::cli {

void print_usage(std::ostream& output);
CliArguments parse_cli_arguments(int argc, char** argv);

}  // namespace tgpu::cli
