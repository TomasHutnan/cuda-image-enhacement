#pragma once

#include <cstddef>
#include <string>

#include "tgpu/pipeline.hpp"

struct GLFWwindow;
struct cudaGraphicsResource;

namespace tgpu::stream {

class GlCudaPresenter {
public:
    explicit GlCudaPresenter(const char* window_title);
    GlCudaPresenter(const GlCudaPresenter&) = delete;
    GlCudaPresenter& operator=(const GlCudaPresenter&) = delete;
    ~GlCudaPresenter();

    void present(const tgpu::DeviceImageF32& image);
    void poll_events();
    bool should_close() const;

private:
    void ensure_initialized(int width, int height);
    void recreate_resources(int width, int height);
    void destroy_resources();
    void draw_frame();

    GLFWwindow* window_ = nullptr;
    const char* window_title_ = nullptr;
    unsigned int texture_ = 0;
    unsigned int pbo_ = 0;
    unsigned int vao_ = 0;
    unsigned int vbo_ = 0;
    unsigned int shader_program_ = 0;
    cudaGraphicsResource* cuda_pbo_resource_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    bool glfw_initialized_ = false;
};

void launch_f32_to_rgba_kernel(const float* src,
                               int width,
                               int height,
                               int src_stride,
                               unsigned char* dst_rgba);

}  // namespace tgpu::stream
