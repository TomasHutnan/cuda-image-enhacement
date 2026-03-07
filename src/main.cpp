#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "tgpu/image_io.hpp"
#include "tgpu/pipeline.hpp"

namespace {

void print_usage() {
    std::cout << "Usage: tgpu_cli <input> <output> [--output-depth u8|u16] [--dump-stages <directory>]\n";
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

        print_usage();
        return 1;
    }

    try {
        const tgpu::ImageF32 input = tgpu::load_grayscale_image(input_path);
        const tgpu::PipelineRunOptions options{.capture_intermediate_stages = dump_stages};
        const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);
        tgpu::save_grayscale_image(output_path, result.output, output_depth);

        if (dump_stages) {
            std::filesystem::create_directories(stages_output_dir);
            for (std::size_t index = 0; index < result.stages.size(); ++index) {
                const tgpu::PipelineStage& stage = result.stages[index];
                const std::filesystem::path stage_path = stages_output_dir /
                    (std::to_string(index) + "_" + sanitize_stage_name(stage.name) + ".png");
                tgpu::save_grayscale_image(stage_path, stage.image, output_depth);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}
