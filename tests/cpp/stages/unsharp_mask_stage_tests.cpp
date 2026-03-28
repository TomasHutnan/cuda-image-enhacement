#include <algorithm>
#include <cstddef>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "tgpu/pipeline.hpp"

namespace
{
    tgpu::PipelineRunOptions unsharp_only_options()
    {
        tgpu::PipelineRunOptions options;
        options.stage_execution.non_local_means = false;
        options.stage_execution.unsharp_mask = true;
        options.stage_execution.richardson_lucy = false;
        options.stage_execution.histogram_stretch = false;
        return options;
    }
}

TEST_CASE("Unsharp mask preserves constant image", "[gpu][pipeline][unsharp]")
{
    tgpu::ImageF32 input;
    input.width = 8;
    input.height = 8;
    input.stride = 8;
    input.data.assign(64, 0.5F);

    tgpu::PipelineRunOptions options = unsharp_only_options();
    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.width == input.width);
    REQUIRE(result.output.height == input.height);

    for (const float value : result.output.data)
    {
        CHECK(value == Catch::Approx(0.5F).margin(1.0e-4F));
    }
}

TEST_CASE("Unsharp mask with zero amount is identity", "[gpu][pipeline][unsharp]")
{
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {
        0.0F, 0.2F, 0.4F,
        0.1F, 0.3F, 0.5F,
        0.6F, 0.8F, 1.0F,
    };

    tgpu::PipelineRunOptions options = unsharp_only_options();
    options.unsharp_mask.amount = 0.0F;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i)
    {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]).margin(1.0e-4F));
    }
}

TEST_CASE("Unsharp mask clamps output to unit interval", "[gpu][pipeline][unsharp]")
{
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {
        0.0F, 0.1F, 0.2F,
        0.3F, 0.4F, 0.5F,
        0.8F, 0.9F, 1.0F,
    };

    tgpu::PipelineRunOptions options = unsharp_only_options();
    options.unsharp_mask.amount = 5.0F;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    for (const float value : result.output.data)
    {
        CHECK(value >= 0.0F);
        CHECK(value <= 1.0F);
    }
}

TEST_CASE("Unsharp mask with invalid sigma falls back to passthrough", "[gpu][pipeline][unsharp]")
{
    tgpu::ImageF32 input;
    input.width = 3;
    input.height = 3;
    input.stride = 3;
    input.data = {
        0.0F, 0.2F, 0.4F,
        0.1F, 0.3F, 0.5F,
        0.6F, 0.8F, 1.0F,
    };

    tgpu::PipelineRunOptions options = unsharp_only_options();
    options.unsharp_mask.sigma = 0.0F;

    const tgpu::PipelineRunResult result = tgpu::run_pipeline(input, options);

    REQUIRE(result.output.data.size() == input.data.size());
    for (std::size_t i = 0; i < input.data.size(); ++i)
    {
        CHECK(result.output.data[i] == Catch::Approx(input.data[i]).margin(1.0e-4F));
    }
}
