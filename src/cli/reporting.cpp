#include "cli/reporting.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace tgpu::cli {
namespace {

struct AggregatedBenchmark {
    double host_to_device_ms = 0.0;
    double non_local_means_ms = 0.0;
    double unsharp_mask_ms = 0.0;
    double richardson_lucy_ms = 0.0;
    double histogram_stretch_ms = 0.0;
    double device_to_host_ms = 0.0;
};

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

std::string format_stage_file_name(const tgpu::PipelineStage& stage) {
    std::ostringstream builder;
    builder << std::setw(2) << std::setfill('0') << stage.prefix << "_" << sanitize_stage_name(stage.name) << ".png";
    return builder.str();
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

}  // namespace tgpu::cli
