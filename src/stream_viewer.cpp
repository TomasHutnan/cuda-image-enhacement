#include "stream_viewer.hpp"

#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace tgpu::stream {

namespace {

constexpr const char* kWindowTitle = "TESCAN GPU Stream";
constexpr std::size_t kScheduleQueueCapacity = 16;
constexpr std::size_t kLoadedQueueCapacity = 8;
constexpr std::size_t kProcessedQueueCapacity = 4;
constexpr std::size_t kDisplayQueueCapacity = 4;
constexpr std::size_t kImageCacheCapacity = 128;
constexpr std::size_t kPrefetchAhead = 64;
constexpr std::chrono::milliseconds kDisplayPollTimeout{5};
constexpr std::chrono::milliseconds kUiPollInterval{1};
constexpr std::size_t kStatusUpdateEveryNFrames = 30;

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

void print_usage(std::ostream& output) {
    output
        << "Usage: tgpu_stream_viewer <input-directory> [OPTIONS]\n"
        << "\n"
        << "  <input-directory>                        Directory with grayscale images\n"
        << "\n"
        << "Options:\n"
        << "  --fps <value>                            Target stream FPS (default: 10.0)\n"
        << "  --side-by-side                           Show original and enhanced images side by side\n"
        << "  --only-stage non_local_means|unsharp_mask|richardson_lucy|histogram_stretch\n"
        << "                                           Process only the specified stage (default: all stages)\n"
        << "  --unsharp-sigma <value>                  Sigma for unsharp mask (default: 1.6666667)\n"
        << "  --unsharp-amount <value>                 Amount for unsharp mask (default: 0.6)\n"
        << "  --rl-iterations <int>                    Number of Richardson-Lucy iterations (default: 2)\n"
        << "  --rl-psf-sigma <value>                   Sigma of the PSF Gaussian kernel (default: 2.5)\n"
        << "  --rl-psf-radius <int>                    Radius of the PSF kernel (default: 7)\n"
        << "  --rl-epsilon <value>                     Convergence threshold for RL (default: 1e-7)\n"
        << "  --strict-kernel-sync                     Synchronize kernels strictly (default: disabled)\n"
        << "  --benchmark                              Print per-frame timings from the GPU pipeline\n"
        << "\n"
        << "Controls:\n"
        << "  q or Esc                                 Exit\n";
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

double parse_double_option(const std::string& value, std::string_view name) {
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(value, &consumed);
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

StreamArguments parse_arguments(int argc, char** argv) {
    if (argc < 2) {
        throw std::invalid_argument("usage");
    }

    StreamArguments arguments;
    arguments.input_directory = argv[1];

    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--fps" && index + 1 < argc) {
            arguments.fps = parse_double_option(argv[++index], "fps");
            continue;
        }
        if (argument == "--side-by-side") {
            arguments.side_by_side = true;
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

    if (!std::filesystem::is_directory(arguments.input_directory)) {
        throw std::runtime_error("Input path must be a directory: " + arguments.input_directory.string());
    }

    if (arguments.fps <= 0.0) {
        throw std::runtime_error("--fps must be greater than zero");
    }

    return arguments;
}

cv::Mat image_f32_to_display_mat(const tgpu::ImageF32& image) {
    cv::Mat f32_mat(image.height, image.width, CV_32F);
    for (int row = 0; row < image.height; ++row) {
        const float* src = image.row_ptr(row);
        float* dst = f32_mat.ptr<float>(row);
        std::memcpy(dst, src, image.width * sizeof(float));
    }

    cv::Mat display_u8;
    f32_mat.convertTo(display_u8, CV_8U, 255.0);
    return display_u8;
}

cv::Mat imagegray_to_display_mat(const tgpu::ImageGray& image) {
    cv::Mat mat(image.height, image.width, CV_8U);
    for (int row = 0; row < image.height; ++row) {
        const uint8_t* src = image.row_ptr(row);
        uint8_t* dst = mat.ptr<uint8_t>(row);
        std::memcpy(dst, src, image.width * sizeof(uint8_t));
    }
    return mat;
}

cv::Mat create_side_by_side_display(const tgpu::ImageGray& original, const tgpu::ImageF32& processed) {
    cv::Mat original_mat = imagegray_to_display_mat(original);
    cv::Mat processed_mat = image_f32_to_display_mat(processed);

    // Create composite image with both side by side
    cv::Mat composite(original_mat.rows, original_mat.cols * 2 + 10, CV_8U, cv::Scalar(0));
    original_mat.copyTo(composite(cv::Rect(0, 0, original_mat.cols, original_mat.rows)));
    processed_mat.copyTo(composite(cv::Rect(original_mat.cols + 10, 0, processed_mat.cols, processed_mat.rows)));

    return composite;
}

void print_status_line(std::ostream& output,
                       double actual_fps,
                       double load_ms,
                       double pipeline_ms,
                       double convert_ms,
                       double show_ms,
                       std::size_t schedule_queue_size,
                       std::size_t loaded_queue_size,
                       std::size_t processed_queue_size,
                       std::size_t display_queue_size,
                       const tgpu::PipelineBenchmark& benchmark) {
    output << '\r' << std::fixed << std::setprecision(1)
           << "Actual FPS: " << actual_fps
           << " (load=" << load_ms << "ms"
           << " pipe=" << pipeline_ms << "ms"
           << " conv=" << convert_ms << "ms"
           << " show=" << show_ms << "ms"
           << " q=" << schedule_queue_size << '/' << loaded_queue_size << '/'
           << processed_queue_size << '/' << display_queue_size << ')';

    if (benchmark.collected) {
        output << " [h2d=" << benchmark.host_to_device_ms
               << " nlm=" << benchmark.non_local_means_ms
               << " usm=" << benchmark.unsharp_mask_ms
               << " rl=" << benchmark.richardson_lucy_ms
               << " hist=" << benchmark.histogram_stretch_ms
               << " d2h=" << benchmark.device_to_host_ms << ']';
    }

    output << "   " << std::flush;
}

}  // namespace

int run_stream_viewer(int argc, char** argv) {
    std::atomic<bool> stop_requested = false;
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    std::mutex prefetch_mutex;
    std::condition_variable prefetch_cv;
    std::size_t latest_scheduled_frame_index = 0;

    try {
        const StreamArguments arguments = parse_arguments(argc, argv);
        const auto image_files = list_image_files(arguments.input_directory);
        if (image_files.empty()) {
            std::cerr << "No supported image files found in directory: " << arguments.input_directory << '\n';
            return 1;
        }

        const auto frame_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(1.0 / arguments.fps));
        ImageCache image_cache(kImageCacheCapacity);

        BoundedQueue<ScheduledFrame> schedule_queue(kScheduleQueueCapacity);
        BoundedQueue<LoadedFrame> loaded_queue(kLoadedQueueCapacity);
        BoundedQueue<ProcessedFrame> processed_queue(kProcessedQueueCapacity);
        BoundedQueue<DisplayFrame> display_queue(kDisplayQueueCapacity);

        auto fail_workers = [&](std::exception_ptr error) {
            {
                std::lock_guard<std::mutex> lock(worker_error_mutex);
                if (!worker_error) {
                    worker_error = error;
                }
            }
            stop_requested = true;
            schedule_queue.close();
            loaded_queue.close();
            processed_queue.close();
            display_queue.close();
            prefetch_cv.notify_all();
        };

        std::thread scheduler_thread([&] {
            try {
                std::size_t frame_index = 0;
                auto next_deadline = std::chrono::steady_clock::now();
                while (!stop_requested.load()) {
                    const std::filesystem::path& path = image_files[frame_index % image_files.size()];
                    if (!schedule_queue.push_drop_oldest(ScheduledFrame{frame_index, path})) {
                        break;
                    }
                    {
                        std::lock_guard<std::mutex> lock(prefetch_mutex);
                        latest_scheduled_frame_index = frame_index + 1;
                    }
                    prefetch_cv.notify_one();
                    ++frame_index;

                    next_deadline += frame_period;
                    const auto now = std::chrono::steady_clock::now();
                    if (now < next_deadline) {
                        std::this_thread::sleep_for(next_deadline - now);
                    } else {
                        next_deadline = now;
                    }
                }
            } catch (...) {
                fail_workers(std::current_exception());
            }
            schedule_queue.close();
        });

        std::thread loader_thread([&] {
            try {
                ScheduledFrame scheduled;
                while (schedule_queue.pop_wait(scheduled, stop_requested)) {
                    const auto load_started = std::chrono::steady_clock::now();
                    tgpu::ImageGray input = image_cache.get(scheduled.image_path);
                    const auto load_finished = std::chrono::steady_clock::now();
                    const double load_ms = std::chrono::duration<double, std::milli>(
                                               load_finished - load_started)
                                               .count();

                    if (!loaded_queue.push_drop_oldest(LoadedFrame{scheduled.frame_index, std::move(input), load_ms})) {
                        break;
                    }
                }
            } catch (...) {
                fail_workers(std::current_exception());
            }
            loaded_queue.close();
        });

        std::thread prefetch_thread([&] {
            try {
                std::size_t next_prefetch_index = 0;
                while (!stop_requested.load()) {
                    std::size_t target_index = 0;
                    {
                        std::unique_lock<std::mutex> lock(prefetch_mutex);
                        prefetch_cv.wait(lock, [&] {
                            return stop_requested.load() || latest_scheduled_frame_index > next_prefetch_index;
                        });
                        if (stop_requested.load()) {
                            break;
                        }
                        target_index = latest_scheduled_frame_index + kPrefetchAhead;
                    }

                    while (!stop_requested.load() && next_prefetch_index < target_index) {
                        const std::filesystem::path& path = image_files[next_prefetch_index % image_files.size()];
                        image_cache.prefetch(path);
                        ++next_prefetch_index;
                    }
                }
            } catch (...) {
                fail_workers(std::current_exception());
            }
        });

        std::thread gpu_thread([&] {
            bool batch_initialized = false;
            int current_width = 0;
            int current_height = 0;
            try {
                LoadedFrame loaded;
                while (loaded_queue.pop_wait(loaded, stop_requested)) {
                    if (!batch_initialized) {
                        tgpu::begin_pipeline_batch(loaded.input.width, loaded.input.height);
                        batch_initialized = true;
                        current_width = loaded.input.width;
                        current_height = loaded.input.height;
                    } else if (loaded.input.width != current_width || loaded.input.height != current_height) {
                        tgpu::end_pipeline_batch();
                        tgpu::begin_pipeline_batch(loaded.input.width, loaded.input.height);
                        current_width = loaded.input.width;
                        current_height = loaded.input.height;
                    }

                    const auto pipeline_started = std::chrono::steady_clock::now();
                    tgpu::PipelineRunResult result = tgpu::run_pipeline(loaded.input, arguments.pipeline_options);
                    const auto pipeline_finished = std::chrono::steady_clock::now();
                    const double pipeline_ms = std::chrono::duration<double, std::milli>(
                                                   pipeline_finished - pipeline_started)
                                                   .count();

                    if (!processed_queue.push_drop_oldest(
                            ProcessedFrame{loaded.frame_index, loaded.input, std::move(result.output), result.benchmark,
                                           loaded.load_ms, pipeline_ms})) {
                        break;
                    }
                }
            } catch (...) {
                fail_workers(std::current_exception());
            }

            if (batch_initialized) {
                try {
                    tgpu::end_pipeline_batch();
                } catch (...) {
                }
            }
            processed_queue.close();
        });

        std::thread convert_thread([&] {
            try {
                ProcessedFrame processed;
                while (processed_queue.pop_wait(processed, stop_requested)) {
                    const auto convert_started = std::chrono::steady_clock::now();

                    cv::Mat display;
                    if (arguments.side_by_side) {
                        display = create_side_by_side_display(processed.original, processed.output);
                    } else {
                        display = image_f32_to_display_mat(processed.output);
                    }

                    const auto convert_finished = std::chrono::steady_clock::now();
                    const double convert_ms = std::chrono::duration<double, std::milli>(
                                                  convert_finished - convert_started)
                                                  .count();

                    if (!display_queue.push_drop_oldest(
                            DisplayFrame{processed.frame_index, std::move(display), processed.benchmark,
                                         processed.load_ms, processed.pipeline_ms, convert_ms})) {
                        break;
                    }
                }
            } catch (...) {
                fail_workers(std::current_exception());
            }
            display_queue.close();
        });

        std::cout << "Starting stream from " << arguments.input_directory
                  << " with " << image_files.size() << " image(s) at "
                  << arguments.fps << " FPS. Press q or Esc to exit.\n";

        cv::namedWindow(kWindowTitle, cv::WINDOW_NORMAL);

        auto loop_started = std::chrono::steady_clock::now();
        std::size_t displayed_frames = 0;
        double last_load_ms = 0.0;
        double last_pipeline_ms = 0.0;
        double last_convert_ms = 0.0;
        double last_imshow_ms = 0.0;
        tgpu::PipelineBenchmark last_benchmark{};

        while (!stop_requested.load()) {
            DisplayFrame frame;
            if (display_queue.pop_wait_for(frame, kDisplayPollTimeout, stop_requested)) {
                const auto imshow_started = std::chrono::steady_clock::now();
                cv::imshow(kWindowTitle, frame.display);
                const auto imshow_finished = std::chrono::steady_clock::now();

                ++displayed_frames;
                last_load_ms = frame.load_ms;
                last_pipeline_ms = frame.pipeline_ms;
                last_convert_ms = frame.convert_ms;
                last_imshow_ms = std::chrono::duration<double, std::milli>(imshow_finished - imshow_started).count();
                last_benchmark = frame.benchmark;
            }

            const int key = cv::waitKey(static_cast<int>(kUiPollInterval.count()));
            if (key == 27 || key == 'q' || key == 'Q') {
                stop_requested = true;
                break;
            }

            if (displayed_frames > 0 && displayed_frames % kStatusUpdateEveryNFrames == 0) {
                const auto now = std::chrono::steady_clock::now();
                const double elapsed_s = std::chrono::duration<double>(now - loop_started).count();
                const double actual_fps = elapsed_s > 0.0 ? static_cast<double>(displayed_frames) / elapsed_s : 0.0;
                print_status_line(std::cout,
                                  actual_fps,
                                  last_load_ms,
                                  last_pipeline_ms,
                                  last_convert_ms,
                                  last_imshow_ms,
                                  schedule_queue.size(),
                                  loaded_queue.size(),
                                  processed_queue.size(),
                                  display_queue.size(),
                                  last_benchmark);
            }
        }

        stop_requested = true;
        schedule_queue.close();
        loaded_queue.close();
        processed_queue.close();
        display_queue.close();

        scheduler_thread.join();
        loader_thread.join();
        prefetch_thread.join();
        gpu_thread.join();
        convert_thread.join();

        {
            std::lock_guard<std::mutex> lock(worker_error_mutex);
            if (worker_error) {
                std::rethrow_exception(worker_error);
            }
        }

        std::cout << '\n';
        cv::destroyAllWindows();
        return 0;
    } catch (const std::invalid_argument&) {
        print_usage(std::cout);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
    }

    return 1;
}

}  // namespace tgpu::stream

int main(int argc, char** argv) {
    return tgpu::stream::run_stream_viewer(argc, argv);
}