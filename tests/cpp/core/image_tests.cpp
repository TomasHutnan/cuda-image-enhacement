#include <cstddef>

#include <catch2/catch_test_macros.hpp>

#include "tgpu/image.hpp"

TEST_CASE("ImageF32 reports empty state consistently", "[image]") {
    tgpu::ImageF32 image;
    REQUIRE(image.empty());

    image.width = 4;
    image.height = 3;
    image.stride = 4;
    image.data.resize(12, 0.0F);

    REQUIRE_FALSE(image.empty());
    REQUIRE(image.size() == 12);
}
