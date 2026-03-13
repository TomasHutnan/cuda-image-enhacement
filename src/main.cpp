#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include "tgpu/image_io.hpp"
#include "tgpu/pipeline.hpp"

namespace {

void print_usage() {
    std::cout << "Usage: tgpu_cli <input> <output> [--output-depth u8|u16] [--dump-stages <directory>]"
                 " [--only-stage non_local_means|unsharp_mask|richardson_lucy|histogram_stretch]\n";
}

tgpu::BitDepth parse_bit_depth(const std::string& value) {
    if (value == "u8") {
        return tgpu::BitDepth::u8;
    }
    if (value == "u16") {
        return tgpu::BitDepth::u16;
    }
    throw std::runtime_error("Unsupported output depth: " + value);
}

std::string sanitize_stage_name(std::string_view value) {
    std::string sanitized;
    sanitized.reserve(value.size());

    for (const char character : value) {
        if ((character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-') {
            sanitized.push_back(character);
        } else {
            sanitized.push_back('_');
        }
    }

    return sanitized;
}

std::string format_stage_file_name(const tgpu::PipelineStage& stage) {
    std::ostringstream builder;
    builder << std::setw(2) << std::setfill('0') << stage.prefix << "_" << sanitize_stage_name(stage.name) << ".png";
    return builder.str();
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

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage();
        return 1;
    }

    const std::filesystem::path input_path = argv[1];
    const std::filesystem::path output_path = argv[2];

    tgpu::BitDepth output_depth = tgpu::BitDepth::u8;
    std::filesystem::path stages_output_dir;
    bool dump_stages = false;
    tgpu::PipelineRunOptions options;

    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--output-depth" && index + 1 < argc) {
            output_depth = parse_bit_depth(argv[++index]);
            continue;
        }
        if (argument == "--dump-stages" && index + 1 < argc) {
            dump_stages = true;
            stages_output_dir = argv[++index];
            continue;
        }
        if (argument == "--only-stage" && index + 1 < argc) {
            apply_only_stage_option(options, argv[++index]);
            continue;
        }

        print_usage();
        return 1;
    }

    try {
        const tgpu::ImageGray input = tgpu::load_grayscale_image_raw(input_path);
        options.capture_intermediate_stages = dump_stages;
        const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);
        tgpu::save_grayscale_image(output_path, result.output, output_depth);

        if (dump_stages) {
            std::filesystem::create_directories(stages_output_dir);
            for (const tgpu::PipelineStage& stage : result.stages) {
                const std::filesystem::path stage_path = stages_output_dir / format_stage_file_name(stage);
                tgpu::save_grayscale_image(stage_path, stage.image, output_depth);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
