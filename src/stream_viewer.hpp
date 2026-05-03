#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "tgpu/image_io.hpp"
#include "tgpu/pipeline.hpp"

namespace tgpu::stream {

enum class DisplayBackend {
	opengl,
	opencv,
};

struct ScheduledFrame {
	std::size_t frame_index = 0;
	std::filesystem::path image_path;
};

struct LoadedFrame {
	std::size_t frame_index = 0;
	tgpu::ImageGray input;
	double load_ms = 0.0;
};

struct ProcessedFrame {
	std::size_t frame_index = 0;
	tgpu::ImageGray original;
	tgpu::ImageF32 output;
	tgpu::DeviceImageF32 device_output;
	bool has_device_output = false;
	tgpu::PipelineBenchmark benchmark;
	double load_ms = 0.0;
	double pipeline_ms = 0.0;
};

struct DisplayFrame {
	std::size_t frame_index = 0;
	cv::Mat display;
	tgpu::PipelineBenchmark benchmark;
	double load_ms = 0.0;
	double pipeline_ms = 0.0;
	double convert_ms = 0.0;
};

struct StreamArguments {
	std::filesystem::path input_directory;
	double fps = 10.0;
	bool side_by_side = false;
	DisplayBackend display_backend = DisplayBackend::opengl;
	tgpu::PipelineRunOptions pipeline_options;
};

// Small LRU cache for decoded grayscale images.
class ImageCache {
public:
	explicit ImageCache(std::size_t max_cached_images = 10)
		: max_cached_(max_cached_images) {}

	tgpu::ImageGray get(const std::filesystem::path& path) {
		const std::string key = path.string();

		{
			std::lock_guard<std::mutex> lock(mutex_);
			const auto it = cache_.find(key);
			if (it != cache_.end()) {
				touch_locked(key);
				return it->second;
			}
		}

		tgpu::ImageGray image = tgpu::load_grayscale_image_raw(path);

		{
			std::lock_guard<std::mutex> lock(mutex_);
			const auto [it, inserted] = cache_.try_emplace(key, std::move(image));
			if (!inserted) {
				touch_locked(key);
				return it->second;
			}

			access_order_.push_back(key);
			evict_if_needed_locked();
			return it->second;
		}
	}

	void prefetch(const std::filesystem::path& path) {
		const std::string key = path.string();

		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (cache_.find(key) != cache_.end()) {
				touch_locked(key);
				return;
			}
		}

		tgpu::ImageGray image = tgpu::load_grayscale_image_raw(path);

		{
			std::lock_guard<std::mutex> lock(mutex_);
			if (cache_.find(key) != cache_.end()) {
				touch_locked(key);
				return;
			}

			cache_.emplace(key, std::move(image));
			access_order_.push_back(key);
			evict_if_needed_locked();
		}
	}

private:
	void touch_locked(const std::string& key) {
		const auto it = std::find(access_order_.begin(), access_order_.end(), key);
		if (it != access_order_.end()) {
			access_order_.erase(it);
		}
		access_order_.push_back(key);
	}

	void evict_if_needed_locked() {
		while (cache_.size() > max_cached_ && !access_order_.empty()) {
			const std::string oldest = access_order_.front();
			access_order_.erase(access_order_.begin());
			cache_.erase(oldest);
		}
	}

	mutable std::mutex mutex_;
	std::unordered_map<std::string, tgpu::ImageGray> cache_;
	std::vector<std::string> access_order_;
	std::size_t max_cached_;
};

template <typename T>
class BoundedQueue {
public:
	explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

	bool push_drop_oldest(T item) {
		std::lock_guard<std::mutex> lock(mutex_);
		if (closed_) {
			return false;
		}
		if (queue_.size() >= capacity_) {
			queue_.pop_front();
		}
		queue_.push_back(std::move(item));
		cv_.notify_one();
		return true;
	}

	bool pop_wait_for(T& out, std::chrono::milliseconds timeout, const std::atomic<bool>& stop_requested) {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait_for(lock, timeout, [&] { return closed_ || !queue_.empty() || stop_requested.load(); });
		if (queue_.empty()) {
			return false;
		}
		out = std::move(queue_.front());
		queue_.pop_front();
		return true;
	}

	bool pop_wait(T& out, const std::atomic<bool>& stop_requested) {
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(lock, [&] { return closed_ || !queue_.empty() || stop_requested.load(); });
		if (queue_.empty()) {
			return false;
		}
		out = std::move(queue_.front());
		queue_.pop_front();
		return true;
	}

	std::size_t size() const {
		std::lock_guard<std::mutex> lock(mutex_);
		return queue_.size();
	}

	void close() {
		std::lock_guard<std::mutex> lock(mutex_);
		closed_ = true;
		cv_.notify_all();
	}

private:
	std::size_t capacity_;
	mutable std::mutex mutex_;
	std::condition_variable cv_;
	std::deque<T> queue_;
	bool closed_ = false;
};

// Runs the folder-backed GPU stream viewer application.
int run_stream_viewer(int argc, char** argv);

}  // namespace tgpu::stream