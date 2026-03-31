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
    output
        << "Usage: tgpu_cli <input> <output> [OPTIONS]\n"
        << "\n"
        << "  <input>              Input file or directory (supported: .png, .jpg, .jpeg, .tif, .tiff, .bmp, .pgm)\n"
        << "  <output>             Output file or directory (see <input> for supported formats)\n"
        << "\n"
        << "Options:\n"
        << "  --output-depth u8|u16                    Output pixel depth (default: u16)\n"
        << "  --dump-stages <directory>                Save intermediate stages to directory\n"
        << "  --only-stage non_local_means|unsharp_mask|richardson_lucy|histogram_stretch\n"
        << "                                           Process only the specified stage (default: all stages)\n"
        << "  --unsharp-sigma <value>                  Sigma for unsharp mask (default: 1.6666667)\n"
        << "  --unsharp-amount <value>                 Amount for unsharp mask (default: 0.6)\n"
        << "  --rl-iterations <int>                    Number of Richardson-Lucy iterations (default: 2)\n"
        << "  --rl-psf-sigma <value>                   Sigma of the PSF Gaussian kernel (default: 2.5)\n"
        << "  --rl-psf-radius <int>                    Radius of the PSF kernel (default: 7)\n"
        << "  --rl-epsilon <value>                     Convergence threshold for RL (default: 1e-7)\n"
        << "  --strict-kernel-sync                     Synchronize kernels strictly (default: disabled)\n"
        << "  --benchmark                              Run in benchmark mode (default: disabled)\n"
        << "\n";
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
