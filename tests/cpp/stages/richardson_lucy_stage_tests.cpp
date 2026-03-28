#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tgpu/pipeline.hpp"

namespace {
    tgpu::PipelineRunOptions rl_only_options() {
        tgpu::PipelineRunOptions options;
        options.stage_execution.non_local_means = false;
        options.stage_execution.unsharp_mask = false;
        options.stage_execution.richardson_lucy = true;
        options.stage_execution.histogram_stretch = false;
        return options;
    }
}

TEST_CASE("Richardson-Lucy iterations zero passes through unchanged", "[pipeline][richardson_lucy]") {
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {
        0.0F, 0.2F, 0.4F,
        0.1F, 0.3F, 0.5F,
        0.6F, 0.8F, 1.0F,
    };

    tgpu::PipelineRunOptions options = rl_only_options();
    options.richardson_lucy.iterations = 0;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]).margin(1.0e-4F));
    }
}

TEST_CASE("Richardson-Lucy invalid sigma passes through unchanged", "[pipeline][richardson_lucy]") {
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {
        0.1F, 0.2F, 0.3F,
        0.2F, 0.3F, 0.4F,
        0.3F, 0.4F, 0.5F,
    };

    tgpu::PipelineRunOptions options = rl_only_options();
    options.richardson_lucy.psf_sigma = 0.0F;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i) {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]).margin(1.0e-4F));
    }
}

TEST_CASE("Richardson-Lucy preserves constant image", "[pipeline][richardson_lucy]") {
    tgpu::ImageF32 input;
    input.width = 6;
    input.height = 6;
    input.stride = 6;
    input.data.assign(36, 0.5F);

    tgpu::PipelineRunOptions options = rl_only_options();

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (float value : result.output.data) {
        CHECK(value == Catch::Approx(0.5F).margin(2.0e-3F));
    }
}

TEST_CASE("Richardson-Lucy output is non-negative", "[pipeline][richardson_lucy]") {
    tgpu::ImageF32 input;
    input.width = 5;
    input.height = 5;
    input.stride = 5;
    for (int i = 0; i < 25; ++i) {
        input.data.push_back(static_cast<float>(i) / 24.0F);
    }

    tgpu::PipelineRunOptions options = rl_only_options();
    options.richardson_lucy.iterations = 4;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE_FALSE(result.output.data.empty());
    for (float value : result.output.data) {
        CHECK(value >= 0.0F);
    }
}
