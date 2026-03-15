#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tgpu/image.hpp"
#include "tgpu/pipeline.hpp"

namespace {
    tgpu::PipelineRunOptions nlm_only_options() {
        tgpu::PipelineRunOptions options;
        options.stage_execution.non_local_means = true;
        options.stage_execution.unsharp_mask = false;
        options.stage_execution.richardson_lucy = false;
        options.stage_execution.histogram_stretch = false;
        return options;
    }
}

TEST_CASE("Non-local means preserves a constant image", "[pipeline][nlm]") {
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {
        0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F,
        0.5F, 0.5F, 0.5F,
    };

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, nlm_only_options());

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t index = 0; index < input.data.size(); ++index) {
        CHECK(result.output.data[index] == Catch::Approx(input.data[index]));
    }
}

TEST_CASE("Non-local means disabled passes through unchanged", "[pipeline][nlm]") {
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {0.1F, 0.5F, 0.9F, 0.3F, 0.6F, 0.2F, 0.8F, 0.4F, 0.7F};

    tgpu::PipelineRunOptions options = nlm_only_options();
    options.non_local_means.enabled = false;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]));
    }
}

TEST_CASE("Non-local means with zero search radius passes through unchanged", "[pipeline][nlm]") {
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {0.1F, 0.5F, 0.9F, 0.3F, 0.6F, 0.2F, 0.8F, 0.4F, 0.7F};

    tgpu::PipelineRunOptions options = nlm_only_options();
    options.non_local_means.search_radius = 0;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]));
    }
}

TEST_CASE("Non-local means output stays within the unit interval", "[pipeline][nlm]") {
    tgpu::ImageF32 input;
    input.width = 5;
    input.height = 5;
    input.stride = 5;
    for (int i = 0; i < 25; ++i) {
        input.data.push_back(static_cast<float>(i) / 24.0F);
    }

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, nlm_only_options());

    REQUIRE_FALSE(result.output.data.empty());
    for (float value : result.output.data) {
        CHECK(value >= 0.0F);
        CHECK(value <= 1.0F);
    }
}
