#include "cli/cli_options.hpp"

#include <exception>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tgpu::cli {
namespace {

tgpu::BitDepth parse_bit_depth(const std::string& value) {
    if (value == "u8") {
        return tgpu::BitDepth::u8;
    }
    if (value == "u16") {
        return tgpu::BitDepth::u16;
    }
    throw std::runtime_error("Unsupported output depth: " + value);
}

float parse_float_option(const std::string& value, std::string_view name) {
    try {
        std::size_t consumed = 0;
        const float parsed = std::stof(value, &consumed);
        if (consumed != value.size()) {
            throw std::runtime_error("Invalid value for --" + std::string{name} + ": " + value);
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid value for --" + std::string{name} + ": " + value);
    }
}

int parse_int_option(const std::string& value, std::string_view name) {
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != value.size()) {
            throw std::runtime_error("Invalid value for --" + std::string{name} + ": " + value);
        }
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid value for --" + std::string{name} + ": " + value);
    }
}

void apply_only_stage_option(tgpu::PipelineRunOptions& options, const std::string& value) {
    options.stage_execution.non_local_means = false;
    options.stage_execution.unsharp_mask = false;
    options.stage_execution.richardson_lucy = false;
    options.stage_execution.histogram_stretch = false;

    if (value == "non_local_means") {
        options.stage_execution.non_local_means = true;
        return;
    }
    if (value == "unsharp_mask") {
        options.stage_execution.unsharp_mask = true;
        return;
    }
    if (value == "richardson_lucy") {
        options.stage_execution.richardson_lucy = true;
        return;
    }
    if (value == "histogram_stretch") {
        options.stage_execution.histogram_stretch = true;
        return;
    }

    throw std::runtime_error("Unsupported stage name for --only-stage: " + value);
}

}  // namespace

void print_usage(std::ostream& output) {
    output << "Usage: tgpu_cli <input> <output> [--output-depth u8|u16] [--dump-stages <directory>]"
              " [--only-stage non_local_means|unsharp_mask|richardson_lucy|histogram_stretch]"
              " [--unsharp-sigma <value>] [--unsharp-amount <value>]"
              " [--rl-iterations <int>] [--rl-psf-sigma <value>] [--rl-psf-radius <int>] [--rl-epsilon <value>]"
              " [--strict-kernel-sync]"
              " [--benchmark]\n"
              "\nNote: <input> and <output> can be files or directories. When processing directories,\n"
              "all supported image files (.png, .jpg, .jpeg, .tif, .tiff, .bmp, .pgm) are processed.\n";
}

CliArguments parse_cli_arguments(int argc, char** argv) {
    if (argc < 3) {
        throw std::invalid_argument("usage");
    }

    CliArguments arguments;
    arguments.input_path = argv[1];
    arguments.output_path = argv[2];

    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--output-depth" && index + 1 < argc) {
            arguments.output_depth = parse_bit_depth(argv[++index]);
            continue;
        }
        if (argument == "--dump-stages" && index + 1 < argc) {
            arguments.dump_stages = true;
            arguments.stages_output_dir = argv[++index];
            continue;
        }
        if (argument == "--only-stage" && index + 1 < argc) {
            apply_only_stage_option(arguments.pipeline_options, argv[++index]);
            continue;
        }
        if (argument == "--unsharp-sigma" && index + 1 < argc) {
            arguments.pipeline_options.unsharp_mask.sigma = parse_float_option(argv[++index], "unsharp-sigma");
            continue;
        }
        if (argument == "--unsharp-amount" && index + 1 < argc) {
            arguments.pipeline_options.unsharp_mask.amount = parse_float_option(argv[++index], "unsharp-amount");
            continue;
        }
        if (argument == "--rl-iterations" && index + 1 < argc) {
            arguments.pipeline_options.richardson_lucy.iterations = parse_int_option(argv[++index], "rl-iterations");
            continue;
        }
        if (argument == "--rl-psf-sigma" && index + 1 < argc) {
            arguments.pipeline_options.richardson_lucy.psf_sigma = parse_float_option(argv[++index], "rl-psf-sigma");
            continue;
        }
        if (argument == "--rl-psf-radius" && index + 1 < argc) {
            arguments.pipeline_options.richardson_lucy.psf_radius = parse_int_option(argv[++index], "rl-psf-radius");
            continue;
        }
        if (argument == "--rl-epsilon" && index + 1 < argc) {
            arguments.pipeline_options.richardson_lucy.epsilon = parse_float_option(argv[++index], "rl-epsilon");
            continue;
        }
        if (argument == "--benchmark") {
            arguments.pipeline_options.collect_benchmark = true;
            continue;
        }
        if (argument == "--strict-kernel-sync") {
            arguments.pipeline_options.strict_kernel_sync_checks = true;
            continue;
        }

        throw std::invalid_argument("usage");
    }

    return arguments;
}

}  // namespace tgpu::cli
