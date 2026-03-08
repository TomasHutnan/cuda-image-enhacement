#pragma once

#include <filesystem>

#include "tgpu/image.hpp"

namespace tgpu {

ImageGray load_grayscale_image_raw(const std::filesystem::path& path);
ImageF32 load_grayscale_image(const std::filesystem::path& path);
void save_grayscale_image(const std::filesystem::path& path, const ImageF32& image, BitDepth bit_depth);

}  // namespace tgpu
