#include <cstddef>

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

    REQUIRE(result.stages.size() == 2);

    CHECK(result.stages[0].prefix == 0);
    CHECK(result.stages[0].name == "input_normalized");
    CHECK(result.stages[0].image.data == input.data);

    CHECK(result.stages[1].prefix == 90);
    CHECK(result.stages[1].name == "output");
    CHECK(result.stages[1].image.data == input.data);
}