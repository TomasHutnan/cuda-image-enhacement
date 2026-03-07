#pragma once

#include <filesystem>

#include "tgpu/image.hpp"

namespace tgpu {

ImageF32 load_grayscale_image(const std::filesystem::path& path);
void save_grayscale_image(const std::filesystem::path& path, const ImageF32& image, BitDepth bit_depth);

}  // namespace tgpu
