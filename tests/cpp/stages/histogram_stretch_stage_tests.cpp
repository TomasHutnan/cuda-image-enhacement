#include <algorithm>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tgpu/image.hpp"
#include "tgpu/pipeline.hpp"

namespace {
    tgpu::PipelineRunOptions histogram_only_options() {
        tgpu::PipelineRunOptions options;
        options.stage_execution.non_local_means = false;
        options.stage_execution.unsharp_mask = false;
        options.stage_execution.richardson_lucy = false;
        options.stage_execution.histogram_stretch = true;
        return options;
    }
}

TEST_CASE("Histogram stretch passes through a flat image unchanged", "[pipeline][histogram]") {
    tgpu::ImageF32 input;
    input.width = 8;
    input.height = 8;
    input.stride = 8;
    input.data.assign(64, 0.5F);

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, histogram_only_options());

    REQUIRE(result.output.data.size() == input.data.size());
    for (float value : result.output.data) {
        CHECK(value == Catch::Approx(0.5F));
    }
}

TEST_CASE("Histogram stretch disabled passes through unchanged", "[pipeline][histogram]") {
    tgpu::ImageF32 input;
    input.width = 4;
    input.height = 4;
    input.stride = 4;
    input.data = {
        0.0F, 0.1F, 0.2F, 0.3F,
        0.4F, 0.5F, 0.6F, 0.7F,
        0.0F, 0.1F, 0.2F, 0.3F,
        0.4F, 0.5F, 0.6F, 0.7F,
    };

    tgpu::PipelineRunOptions options = histogram_only_options();
    options.histogram_stretch.enabled = false;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]));
    }
}

TEST_CASE("Histogram stretch on full-range input with zero saturation is approximately identity", "[pipeline][histogram]") {
    tgpu::ImageF32 input;
    input.width = 4;
    input.height = 4;
    input.stride = 4;
    input.data = {
        0.0F, 0.2F, 0.4F, 0.6F,
        0.8F, 1.0F, 0.1F, 0.3F,
        0.5F, 0.7F, 0.9F, 0.2F,
        0.4F, 0.6F, 0.8F, 0.0F,
    };

    tgpu::PipelineRunOptions options = histogram_only_options();
    options.histogram_stretch.saturation_percent = 0.0F;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]).margin(0.001F));
    }
}

TEST_CASE("Histogram stretch expands a narrow tonal range toward the unit interval", "[pipeline][histogram]") {
    tgpu::ImageF32 input;
    input.width = 8;
    input.height = 8;
    input.stride = 8;
    for (int i = 0; i < 64; ++i) {
        input.data.push_back((i % 2 == 0) ? 0.25F : 0.75F);
    }

    tgpu::PipelineRunOptions options = histogram_only_options();
    options.histogram_stretch.saturation_percent = 0.0F;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    float min_out = 1.0F;
    float max_out = 0.0F;
    for (float value : result.output.data) {
        min_out = std::min(min_out, value);
        max_out = std::max(max_out, value);
    }

    CHECK(min_out < 0.05F);
    CHECK(max_out > 0.95F);
}

TEST_CASE("Histogram stretch output values stay within the unit interval", "[pipeline][histogram]") {
    tgpu::ImageF32 input;
    input.width = 4;
    input.height = 4;
    input.stride = 4;
    input.data = {
        0.0F, 0.1F, 0.3F, 0.5F,
        0.7F, 0.9F, 1.0F, 0.2F,
        0.4F, 0.6F, 0.8F, 0.15F,
        0.25F, 0.55F, 0.75F, 0.95F,
    };

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, histogram_only_options());

    REQUIRE_FALSE(result.output.data.empty());
    for (float value : result.output.data) {
        CHECK(value >= 0.0F);
        CHECK(value <= 1.0F);
    }
}
