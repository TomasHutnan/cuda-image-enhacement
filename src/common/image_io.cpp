#include "tgpu/image_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace tgpu {

namespace {

float normalize_u8(std::uint8_t value) {
    return static_cast<float>(value) / 255.0F;
}

float normalize_u16(std::uint16_t value) {
    return static_cast<float>(value) / 65535.0F;
}

std::uint8_t denormalize_u8(float value) {
    const float clamped = std::clamp(value, 0.0F, 1.0F);
    return static_cast<std::uint8_t>(std::lround(clamped * 255.0F));
}

std::uint16_t denormalize_u16(float value) {
    const float clamped = std::clamp(value, 0.0F, 1.0F);
    return static_cast<std::uint16_t>(std::lround(clamped * 65535.0F));
}

}  // namespace

ImageGray load_grayscale_image_raw(const std::filesystem::path& path) {
    const cv::Mat image = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image: " + path.string());
    }

    if (image.channels() != 1) {
        throw std::runtime_error("Expected a single-channel grayscale image: " + path.string());
    }

    ImageGray result;
    result.width = image.cols;
    result.height = image.rows;

    if (image.type() == CV_8UC1) {
        result.bit_depth = BitDepth::u8;
        result.stride_bytes = image.cols;
    } else if (image.type() == CV_16UC1) {
        result.bit_depth = BitDepth::u16;
        result.stride_bytes = image.cols * static_cast<int>(sizeof(std::uint16_t));
    } else {
        throw std::runtime_error("Unsupported grayscale image type: " + path.string());
    }

    result.data.resize(static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.stride_bytes));
    for (int row = 0; row < image.rows; ++row) {
        const auto* src = image.ptr<std::uint8_t>(row);
        std::copy_n(src, result.stride_bytes, result.row_ptr(row));
    }

    return result;
}

ImageF32 load_grayscale_image(const std::filesystem::path& path) {
    const ImageGray image = load_grayscale_image_raw(path);

    ImageF32 result;
    result.width = image.width;
    result.height = image.height;
    result.stride = image.width;
    result.data.resize(static_cast<std::size_t>(result.width) * static_cast<std::size_t>(result.height));

    if (image.bit_depth == BitDepth::u8) {
        for (int row = 0; row < image.height; ++row) {
            const auto* src = image.row_ptr(row);
            float* dst = result.row_ptr(row);
            for (int col = 0; col < image.width; ++col) {
                dst[col] = normalize_u8(src[col]);
            }
        }
        return result;
    }

    for (int row = 0; row < image.height; ++row) {
        const auto* src = image.row_ptr(row);
        float* dst = result.row_ptr(row);
        for (int col = 0; col < image.width; ++col) {
            std::uint16_t value = 0;
            std::memcpy(&value, src + static_cast<std::size_t>(col) * sizeof(std::uint16_t), sizeof(value));
            dst[col] = normalize_u16(value);
        }
    }

    return result;
}

void save_grayscale_image(const std::filesystem::path& path, const ImageF32& image, BitDepth bit_depth) {
    if (image.empty()) {
        throw std::runtime_error("Cannot save an empty image");
    }

    std::filesystem::create_directories(path.parent_path());

    if (bit_depth == BitDepth::u8) {
        cv::Mat output(image.height, image.width, CV_8UC1);
        for (int row = 0; row < image.height; ++row) {
            const float* src = image.row_ptr(row);
            auto* dst = output.ptr<std::uint8_t>(row);
            for (int col = 0; col < image.width; ++col) {
                dst[col] = denormalize_u8(src[col]);
            }
        }
        if (!cv::imwrite(path.string(), output)) {
            throw std::runtime_error("Failed to write image: " + path.string());
        }
        return;
    }

    cv::Mat output(image.height, image.width, CV_16UC1);
    for (int row = 0; row < image.height; ++row) {
        const float* src = image.row_ptr(row);
        auto* dst = output.ptr<std::uint16_t>(row);
        for (int col = 0; col < image.width; ++col) {
            dst[col] = denormalize_u16(src[col]);
        }
    }
    if (!cv::imwrite(path.string(), output)) {
        throw std::runtime_error("Failed to write image: " + path.string());
    }
}

}  // namespace tgpu
