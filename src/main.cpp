#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <cctype>
#include <algorithm>

#include "tgpu/image_io.hpp"
#include "tgpu/pipeline.hpp"

namespace {

// Structure to hold results from processing a single image
struct ProcessingResult {
    bool success = false;
    std::filesystem::path input_file;
    tgpu::PipelineRunResult pipeline_result;
    tgpu::PipelineRunOptions options;
};

// Structure to accumulate benchmark statistics across multiple images
struct AggregatedBenchmark {
    int successful_images = 0;
    double host_to_device_ms = 0.0;
    double non_local_means_ms = 0.0;
    double unsharp_mask_ms = 0.0;
    double richardson_lucy_ms = 0.0;
    double histogram_stretch_ms = 0.0;
    double device_to_host_ms = 0.0;
};

void print_usage() {
    std::cout << "Usage: tgpu_cli <input> <output> [--output-depth u8|u16] [--dump-stages <directory>]"
                 " [--only-stage non_local_means|unsharp_mask|richardson_lucy|histogram_stretch]"
                 " [--unsharp-sigma <value>] [--unsharp-amount <value>]"
                 " [--rl-iterations <int>] [--rl-psf-sigma <value>] [--rl-psf-radius <int>] [--rl-epsilon <value>]"
                 " [--benchmark]\n"
                 "\nNote: <input> and <output> can be files or directories. When processing directories,\n"
                 "all supported image files (.png, .jpg, .jpeg, .tif, .tiff, .bmp, .pgm) are processed.\n";
}

// Convert string to lowercase for case-insensitive comparison
std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

// Check if file extension is a supported image format
bool is_supported_image_extension(const std::filesystem::path& file_path) {
    const std::string ext = to_lower(file_path.extension().string());
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" || 
           ext == ".tif" || ext == ".tiff" || ext == ".bmp" || ext == ".pgm";
}

// List all supported image files in a directory (non-recursive)
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

void print_benchmark_report(const tgpu::PipelineRunResult& result, const tgpu::PipelineRunOptions& options) {
    if (!result.benchmark.collected) {
        return;
    }

    const auto stage_time = [](bool enabled, double ms) -> std::string {
        if (!enabled) {
            return "n/a";
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << ms;
        return stream.str();
    };

    const double filters_ms =
        result.benchmark.non_local_means_ms +
        result.benchmark.unsharp_mask_ms +
        result.benchmark.richardson_lucy_ms +
        result.benchmark.histogram_stretch_ms;
    const double copy_ms = result.benchmark.host_to_device_ms + result.benchmark.device_to_host_ms;
    const double total_ms = filters_ms + copy_ms;

    std::cout << "Benchmark (ms):\n";
    std::cout << "  copy_h2d             " << std::fixed << std::setprecision(3) << result.benchmark.host_to_device_ms << "\n";
    std::cout << "  non_local_means      " << stage_time(options.stage_execution.non_local_means, result.benchmark.non_local_means_ms) << "\n";
    std::cout << "  unsharp_mask         " << stage_time(options.stage_execution.unsharp_mask, result.benchmark.unsharp_mask_ms) << "\n";
    std::cout << "  richardson_lucy      " << stage_time(options.stage_execution.richardson_lucy, result.benchmark.richardson_lucy_ms) << "\n";
    std::cout << "  histogram_stretch    " << stage_time(options.stage_execution.histogram_stretch, result.benchmark.histogram_stretch_ms) << "\n";
    std::cout << "  copy_d2h             " << std::fixed << std::setprecision(3) << result.benchmark.device_to_host_ms << "\n";
    std::cout << "  ------------------------------\n";
    std::cout << "  filters_total        " << std::fixed << std::setprecision(3) << filters_ms << "\n";
    std::cout << "  copy_total           " << std::fixed << std::setprecision(3) << copy_ms << "\n";
    std::cout << "  total                " << std::fixed << std::setprecision(3) << total_ms << "\n";
}

void print_mean_benchmark_report(const std::vector<ProcessingResult>& results, const tgpu::PipelineRunOptions& options) {
    int successful_count = 0;
    AggregatedBenchmark aggregated;

    for (const auto& result : results) {
        if (!result.success || !result.pipeline_result.benchmark.collected) {
            continue;
        }
        successful_count++;
        aggregated.host_to_device_ms += result.pipeline_result.benchmark.host_to_device_ms;
        aggregated.non_local_means_ms += result.pipeline_result.benchmark.non_local_means_ms;
        aggregated.unsharp_mask_ms += result.pipeline_result.benchmark.unsharp_mask_ms;
        aggregated.richardson_lucy_ms += result.pipeline_result.benchmark.richardson_lucy_ms;
        aggregated.histogram_stretch_ms += result.pipeline_result.benchmark.histogram_stretch_ms;
        aggregated.device_to_host_ms += result.pipeline_result.benchmark.device_to_host_ms;
    }

    if (successful_count == 0) {
        std::cerr << "No successful images to report benchmark for.\n";
        return;
    }

    const auto stage_time = [successful_count](bool enabled, double ms) -> std::string {
        if (!enabled) {
            return "n/a";
        }

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(3) << (ms / successful_count);
        return stream.str();
    };

    const double mean_host_to_device = aggregated.host_to_device_ms / successful_count;
    const double mean_non_local_means = aggregated.non_local_means_ms / successful_count;
    const double mean_unsharp_mask = aggregated.unsharp_mask_ms / successful_count;
    const double mean_richardson_lucy = aggregated.richardson_lucy_ms / successful_count;
    const double mean_histogram_stretch = aggregated.histogram_stretch_ms / successful_count;
    const double mean_device_to_host = aggregated.device_to_host_ms / successful_count;

    const double filters_ms = mean_non_local_means + mean_unsharp_mask + mean_richardson_lucy + mean_histogram_stretch;
    const double copy_ms = mean_host_to_device + mean_device_to_host;
    const double total_ms = filters_ms + copy_ms;

    std::cout << "\nMean Benchmark across " << successful_count << " image(s) (ms):\n";
    std::cout << "  copy_h2d             " << std::fixed << std::setprecision(3) << mean_host_to_device << "\n";
    std::cout << "  non_local_means      " << stage_time(options.stage_execution.non_local_means, aggregated.non_local_means_ms) << "\n";
    std::cout << "  unsharp_mask         " << stage_time(options.stage_execution.unsharp_mask, aggregated.unsharp_mask_ms) << "\n";
    std::cout << "  richardson_lucy      " << stage_time(options.stage_execution.richardson_lucy, aggregated.richardson_lucy_ms) << "\n";
    std::cout << "  histogram_stretch    " << stage_time(options.stage_execution.histogram_stretch, aggregated.histogram_stretch_ms) << "\n";
    std::cout << "  copy_d2h             " << std::fixed << std::setprecision(3) << mean_device_to_host << "\n";
    std::cout << "  ------------------------------\n";
    std::cout << "  filters_total        " << std::fixed << std::setprecision(3) << filters_ms << "\n";
    std::cout << "  copy_total           " << std::fixed << std::setprecision(3) << copy_ms << "\n";
    std::cout << "  total                " << std::fixed << std::setprecision(3) << total_ms << "\n";
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
        if (argument == "--unsharp-sigma" && index + 1 < argc) {
            options.unsharp_mask.sigma = parse_float_option(argv[++index], "unsharp-sigma");
            continue;
        }
        if (argument == "--unsharp-amount" && index + 1 < argc) {
            options.unsharp_mask.amount = parse_float_option(argv[++index], "unsharp-amount");
            continue;
        }
        if (argument == "--rl-iterations" && index + 1 < argc) {
            options.richardson_lucy.iterations = parse_int_option(argv[++index], "rl-iterations");
            continue;
        }
        if (argument == "--rl-psf-sigma" && index + 1 < argc) {
            options.richardson_lucy.psf_sigma = parse_float_option(argv[++index], "rl-psf-sigma");
            continue;
        }
        if (argument == "--rl-psf-radius" && index + 1 < argc) {
            options.richardson_lucy.psf_radius = parse_int_option(argv[++index], "rl-psf-radius");
            continue;
        }
        if (argument == "--rl-epsilon" && index + 1 < argc) {
            options.richardson_lucy.epsilon = parse_float_option(argv[++index], "rl-epsilon");
            continue;
        }
        if (argument == "--benchmark") {
            options.collect_benchmark = true;
            continue;
        }

        print_usage();
        return 1;
    }

    // Determine if input is a directory or file
    const bool input_is_directory = std::filesystem::is_directory(input_path);
    const bool output_is_directory = std::filesystem::is_directory(output_path);

    if (!input_is_directory) {
        // Single-file mode: process one image and output to file or directory
        try {
            const tgpu::ImageGray input = tgpu::load_grayscale_image_raw(input_path);
            options.capture_intermediate_stages = dump_stages;
            const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

            // Determine output file path
            std::filesystem::path final_output_path = output_path;
            if (output_is_directory) {
                final_output_path = output_path / input_path.filename();
            }

            tgpu::save_grayscale_image(final_output_path, result.output, output_depth);
            print_benchmark_report(result, options);

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

    // Directory mode: process all images in directory
    try {
        // Create output directory if needed
        if (!output_is_directory) {
            std::filesystem::create_directories(output_path);
        }

        // List all supported image files in input directory
        const auto image_files = list_image_files(input_path);
        if (image_files.empty()) {
            std::cerr << "No supported image files found in directory: " << input_path << '\n';
            return 1;
        }

        std::cout << "Processing " << image_files.size() << " image(s) from " << input_path << "...\n\n";

        std::vector<ProcessingResult> results;
        int failed_count = 0;

        for (const auto& image_file : image_files) {
            std::cout << "Processing: " << image_file.filename() << "... ";
            std::cout.flush();

            ProcessingResult result;
            result.input_file = image_file;
            result.options = options;

            try {
                const tgpu::ImageGray input = tgpu::load_grayscale_image_raw(image_file);
                options.capture_intermediate_stages = dump_stages;
                result.pipeline_result = tgpu::run_pipeline(input, options);

                // Determine output file path
                const std::filesystem::path output_file = output_path / image_file.filename();
                tgpu::save_grayscale_image(output_file, result.pipeline_result.output, output_depth);

                if (dump_stages) {
                    // Create per-image stage dump subfolder: <stem>_<ext_no_dot>/
                    const std::string ext_no_dot = image_file.extension().string();
                    std::string ext_no_dot_lower = to_lower(ext_no_dot.substr(1)); // skip the leading '.'
                    const std::string subfolder_name = image_file.stem().string() + "_" + ext_no_dot_lower;
                    const std::filesystem::path stage_dump_dir = stages_output_dir / subfolder_name;
                    std::filesystem::create_directories(stage_dump_dir);

                    for (const tgpu::PipelineStage& stage : result.pipeline_result.stages) {
                        const std::filesystem::path stage_path = stage_dump_dir / format_stage_file_name(stage);
                        tgpu::save_grayscale_image(stage_path, stage.image, output_depth);
                    }
                }

                result.success = true;
                std::cout << "OK\n";
            } catch (const std::exception& error) {
                std::cerr << "FAILED: " << error.what() << '\n';
                result.success = false;
                failed_count++;
            }

            results.push_back(result);
        }

        std::cout << "\n";
        if (options.collect_benchmark) {
            print_mean_benchmark_report(results, options);
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
