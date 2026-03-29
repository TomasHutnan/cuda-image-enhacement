#include "cli/reporting.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace tgpu::cli {
namespace {

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    if (values.size() == 1) {
        return values.front();
    }

    const double position = (static_cast<double>(values.size() - 1) * p);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = (lower + 1 < values.size()) ? (lower + 1) : lower;
    const double fraction = position - static_cast<double>(lower);
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
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

struct MetricSeries {
    std::string name;
    bool enabled = true;
    std::vector<double> samples;
};

void print_metric_table(const std::vector<MetricSeries>& metrics) {
    std::cout << "\nBenchmark Distribution (ms):\n";
    std::cout << "  " << std::left << std::setw(20) << "metric"
              << std::right << std::setw(10) << "mean"
              << std::setw(10) << "Q1"
              << std::setw(10) << "median"
              << std::setw(10) << "Q3" << "\n";
    std::cout << "  " << std::string(60, '-') << "\n";

    for (std::size_t i = 0; i < metrics.size(); ++i) {
        const MetricSeries& metric = metrics[i];
        std::cout << "  " << std::left << std::setw(20) << metric.name;
        if (!metric.enabled) {
            std::cout << std::right << std::setw(10) << "n/a"
                      << std::setw(10) << "n/a"
                      << std::setw(10) << "n/a"
                      << std::setw(10) << "n/a" << "\n";
            continue;
        }

        std::cout << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << mean(metric.samples)
                  << std::setw(10) << percentile(metric.samples, 0.25)
                  << std::setw(10) << percentile(metric.samples, 0.50)
                  << std::setw(10) << percentile(metric.samples, 0.75)
                  << "\n";

        // Print division line after copy_d2h to separate stages from aggregates
        if (metric.name == "copy_d2h") {
            std::cout << "  " << std::string(60, '-') << "\n";
        }
    }
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

    std::vector<double> host_to_device_samples;
    std::vector<double> non_local_means_samples;
    std::vector<double> unsharp_mask_samples;
    std::vector<double> richardson_lucy_samples;
    std::vector<double> histogram_stretch_samples;
    std::vector<double> device_to_host_samples;
    std::vector<double> filters_total_samples;
    std::vector<double> copy_total_samples;
    std::vector<double> total_samples;

    for (const auto& result : results) {
        if (!result.success || !result.pipeline_result.benchmark.collected) {
            continue;
        }
        successful_count++;

        const auto& b = result.pipeline_result.benchmark;

        const double sample_filters_total =
            (options.stage_execution.non_local_means ? b.non_local_means_ms : 0.0) +
            (options.stage_execution.unsharp_mask ? b.unsharp_mask_ms : 0.0) +
            (options.stage_execution.richardson_lucy ? b.richardson_lucy_ms : 0.0) +
            (options.stage_execution.histogram_stretch ? b.histogram_stretch_ms : 0.0);
        const double sample_copy_total = b.host_to_device_ms + b.device_to_host_ms;
        const double sample_total = sample_filters_total + sample_copy_total;

        host_to_device_samples.push_back(b.host_to_device_ms);
        if (options.stage_execution.non_local_means) {
            non_local_means_samples.push_back(b.non_local_means_ms);
        }
        if (options.stage_execution.unsharp_mask) {
            unsharp_mask_samples.push_back(b.unsharp_mask_ms);
        }
        if (options.stage_execution.richardson_lucy) {
            richardson_lucy_samples.push_back(b.richardson_lucy_ms);
        }
        if (options.stage_execution.histogram_stretch) {
            histogram_stretch_samples.push_back(b.histogram_stretch_ms);
        }
        device_to_host_samples.push_back(b.device_to_host_ms);
        filters_total_samples.push_back(sample_filters_total);
        copy_total_samples.push_back(sample_copy_total);
        total_samples.push_back(sample_total);
    }

    if (successful_count == 0) {
        std::cerr << "No successful images to report benchmark for.\n";
        return;
    }

    std::cout << "\nBenchmark summary across " << successful_count << " image(s).";

    std::vector<MetricSeries> metrics;
    metrics.push_back(MetricSeries{"copy_h2d", true, std::move(host_to_device_samples)});
    metrics.push_back(MetricSeries{"non_local_means", options.stage_execution.non_local_means, std::move(non_local_means_samples)});
    metrics.push_back(MetricSeries{"unsharp_mask", options.stage_execution.unsharp_mask, std::move(unsharp_mask_samples)});
    metrics.push_back(MetricSeries{"richardson_lucy", options.stage_execution.richardson_lucy, std::move(richardson_lucy_samples)});
    metrics.push_back(MetricSeries{"histogram_stretch", options.stage_execution.histogram_stretch, std::move(histogram_stretch_samples)});
    metrics.push_back(MetricSeries{"copy_d2h", true, std::move(device_to_host_samples)});
    metrics.push_back(MetricSeries{"filters_total", true, std::move(filters_total_samples)});
    metrics.push_back(MetricSeries{"copy_total", true, std::move(copy_total_samples)});
    metrics.push_back(MetricSeries{"total", true, std::move(total_samples)});

    print_metric_table(metrics);
}

}  // namespace tgpu::cli
