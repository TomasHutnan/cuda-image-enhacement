#include "cli/processing.hpp"

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "cli/reporting.hpp"
#include "tgpu/image_io.hpp"

namespace tgpu::cli {
namespace {

std::string to_lower(const std::string& value) {
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lowered;
}

bool is_supported_image_extension(const std::filesystem::path& file_path) {
    const std::string ext = to_lower(file_path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tif" ||
           ext == ".tiff" || ext == ".bmp" || ext == ".pgm";
}

std::vector<std::filesystem::path> list_image_files(const std::filesystem::path& directory) {
    std::vector<std::filesystem::path> images;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && is_supported_image_extension(entry.path())) {
            images.push_back(entry.path());
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

}  // namespace

int run_single_file_mode(const CliArguments& arguments) {
    try {
        const bool output_is_directory = std::filesystem::is_directory(arguments.output_path);
        const tgpu::ImageGray input = tgpu::load_grayscale_image_raw(arguments.input_path);

        tgpu::PipelineRunOptions options = arguments.pipeline_options;
        options.capture_intermediate_stages = arguments.dump_stages;

        const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

        std::filesystem::path final_output_path = arguments.output_path;
        if (output_is_directory) {
            final_output_path = arguments.output_path / arguments.input_path.filename();
        }

        tgpu::save_grayscale_image(final_output_path, result.output, arguments.output_depth);
        print_benchmark_report(result, options);

        if (arguments.dump_stages) {
            std::filesystem::create_directories(arguments.stages_output_dir);
            for (const tgpu::PipelineStage& stage : result.stages) {
                const std::filesystem::path stage_path = arguments.stages_output_dir / format_stage_file_name(stage);
                tgpu::save_grayscale_image(stage_path, stage.image, arguments.output_depth);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

int run_directory_mode(const CliArguments& arguments) {
    try {
        const bool output_is_directory = std::filesystem::is_directory(arguments.output_path);
        if (!output_is_directory) {
            std::filesystem::create_directories(arguments.output_path);
        }

        const auto image_files = list_image_files(arguments.input_path);
        if (image_files.empty()) {
            std::cerr << "No supported image files found in directory: " << arguments.input_path << '\n';
            return 1;
        }

        std::cout << "Processing " << image_files.size() << " image(s) from " << arguments.input_path << "...\n\n";

        std::vector<ProcessingResult> results;
        int failed_count = 0;

        for (const auto& image_file : image_files) {
            std::cout << "Processing: " << image_file.filename() << "... ";
            std::cout.flush();

            ProcessingResult result;
            result.input_file = image_file;

            try {
                const tgpu::ImageGray input = tgpu::load_grayscale_image_raw(image_file);
                tgpu::PipelineRunOptions options = arguments.pipeline_options;
                options.capture_intermediate_stages = arguments.dump_stages;

                result.pipeline_result = tgpu::run_pipeline(input, options);

                const std::filesystem::path output_file = arguments.output_path / image_file.filename();
                tgpu::save_grayscale_image(output_file, result.pipeline_result.output, arguments.output_depth);

                if (arguments.dump_stages) {
                    const std::string extension = image_file.extension().string();
                    const std::string extension_no_dot = extension.size() > 1 ? extension.substr(1) : "";
                    const std::string subfolder_name = image_file.stem().string() + "_" + to_lower(extension_no_dot);
                    const std::filesystem::path stage_dump_dir = arguments.stages_output_dir / subfolder_name;
                    std::filesystem::create_directories(stage_dump_dir);

                    for (const tgpu::PipelineStage& stage : result.pipeline_result.stages) {
                        const std::filesystem::path stage_path = stage_dump_dir / format_stage_file_name(stage);
                        tgpu::save_grayscale_image(stage_path, stage.image, arguments.output_depth);
                    }
                }

                result.success = true;
                std::cout << "OK\n";
            } catch (const std::exception& error) {
                std::cerr << "FAILED: " << error.what() << '\n';
                result.success = false;
                failed_count++;
            }

            results.push_back(std::move(result));
        }

        std::cout << "\n";
        if (arguments.pipeline_options.collect_benchmark) {
            print_mean_benchmark_report(results, arguments.pipeline_options);
        }

        if (failed_count > 0) {
            std::cout << "\n" << failed_count << " image(s) failed to process.\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    return 0;
}

}  // namespace tgpu::cli
