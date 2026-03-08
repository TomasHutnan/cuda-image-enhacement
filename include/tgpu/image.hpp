#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tgpu {

enum class BitDepth {
    u8,
    u16,
};

struct ImageGray {
    int width = 0;
    int height = 0;
    int stride_bytes = 0;
    BitDepth bit_depth = BitDepth::u8;
    std::vector<std::uint8_t> data;

    [[nodiscard]] std::size_t bytes_per_pixel() const noexcept {
        return bit_depth == BitDepth::u8 ? 1U : 2U;
    }

    [[nodiscard]] bool empty() const noexcept {
        return data.empty() || width <= 0 || height <= 0 ||
               stride_bytes < static_cast<int>(static_cast<std::size_t>(width) * bytes_per_pixel());
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return data.size();
    }

    [[nodiscard]] const std::uint8_t* row_ptr(int row) const noexcept {
        return data.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(stride_bytes);
    }

    [[nodiscard]] std::uint8_t* row_ptr(int row) noexcept {
        return data.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(stride_bytes);
    }
};

struct ImageF32 {
    int width = 0;
    int height = 0;
    int stride = 0;
    std::vector<float> data;

    [[nodiscard]] bool empty() const noexcept {
        return data.empty() || width <= 0 || height <= 0 || stride < width;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return data.size();
    }

    [[nodiscard]] const float* row_ptr(int row) const noexcept {
        return data.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(stride);
    }

    [[nodiscard]] float* row_ptr(int row) noexcept {
        return data.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(stride);
    }
};

}  // namespace tgpu
