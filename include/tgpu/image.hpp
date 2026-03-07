#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tgpu {

enum class BitDepth {
    u8,
    u16,
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
