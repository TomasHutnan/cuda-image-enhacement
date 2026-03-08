#include <cstddef>
#include <cstdint>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tgpu/image.hpp"
#include "tgpu/pipeline.hpp"

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

TEST_CASE("Pipeline passthrough preserves normalized image dimensions", "[pipeline]") {
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 2;
    input.stride = 3;
    input.data = {0.0F, 0.25F, 0.5F, 0.75F, 1.0F, 0.125F};

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input);

    REQUIRE(result.stages.empty());
    REQUIRE(result.output.width == input.width);
    REQUIRE(result.output.height == input.height);
    REQUIRE(result.output.stride == input.stride);
    REQUIRE(result.output.data == input.data);
}

TEST_CASE("Pipeline stage capture uses stable stage names and prefixes", "[pipeline]") {
    tgpu::ImageF32 input;
    input.width = 2;
    input.height = 2;
    input.stride = 2;
    input.data = {0.1F, 0.2F, 0.3F, 0.4F};

    const tgpu::PipelineRunOptions options{.capture_intermediate_stages = true};
    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.stages.size() == 6);

    CHECK(result.stages[0].prefix == 0);
    CHECK(result.stages[0].name == "input_normalized");
    CHECK(result.stages[0].image.data == input.data);

    CHECK(result.stages[1].prefix == 10);
    CHECK(result.stages[1].name == "non_local_means");
    CHECK(result.stages[1].image.data == input.data);

    CHECK(result.stages[2].prefix == 20);
    CHECK(result.stages[2].name == "unsharp_mask");
    CHECK(result.stages[2].image.data == input.data);

    CHECK(result.stages[3].prefix == 30);
    CHECK(result.stages[3].name == "richardson_lucy");
    CHECK(result.stages[3].image.data == input.data);

    CHECK(result.stages[4].prefix == 40);
    CHECK(result.stages[4].name == "histogram_stretch");
    CHECK(result.stages[4].image.data == input.data);

    CHECK(result.stages[5].prefix == 90);
    CHECK(result.stages[5].name == "output");
    CHECK(result.stages[5].image.data == input.data);
}

TEST_CASE("Pipeline normalizes raw grayscale input on the GPU", "[pipeline]") {
    tgpu::ImageGray input;
    input.width = 2;
    input.height = 2;
    input.stride_bytes = 2;
    input.bit_depth = tgpu::BitDepth::u8;
    input.data = {0, 64, 128, 255};

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input);

    REQUIRE(result.output.width == input.width);
    REQUIRE(result.output.height == input.height);
    REQUIRE(result.output.stride == input.width);
    REQUIRE(result.output.data.size() == 4);

    CHECK(result.output.data[0] == Catch::Approx(0.0F));
    CHECK(result.output.data[1] == Catch::Approx(64.0F / 255.0F));
    CHECK(result.output.data[2] == Catch::Approx(128.0F / 255.0F));
    CHECK(result.output.data[3] == Catch::Approx(1.0F));
}