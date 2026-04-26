#include "stream_viewer_gl.hpp"

#include <stdexcept>
#include <string>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/glad.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace tgpu::stream {

namespace {

constexpr const char* kVertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec2 in_uv;
out vec2 frag_uv;
void main() {
    frag_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
)";

constexpr const char* kFragmentShaderSource = R"(
#version 330 core
in vec2 frag_uv;
out vec4 out_color;
uniform sampler2D image_texture;
void main() {
    out_color = texture(image_texture, frag_uv);
}
)";

void throw_cuda_if_failed(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

unsigned int compile_shader(unsigned int type, const char* source)
{
    const unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    int success = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (success == GL_FALSE)
    {
        int length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(length), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error("OpenGL shader compilation failed: " + log);
    }

    return shader;
}

unsigned int build_program()
{
    const unsigned int vertex = compile_shader(GL_VERTEX_SHADER, kVertexShaderSource);
    const unsigned int fragment = compile_shader(GL_FRAGMENT_SHADER, kFragmentShaderSource);

    const unsigned int program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);

    glDeleteShader(vertex);
    glDeleteShader(fragment);

    int success = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (success == GL_FALSE)
    {
        int length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(length), '\0');
        glGetProgramInfoLog(program, length, nullptr, log.data());
        glDeleteProgram(program);
        throw std::runtime_error("OpenGL program link failed: " + log);
    }

    return program;
}

void setup_fullscreen_quad(unsigned int& vao, unsigned int& vbo)
{
    constexpr float vertices[] = {
        -1.0F, -1.0F, 0.0F, 0.0F,
         1.0F, -1.0F, 1.0F, 0.0F,
         1.0F,  1.0F, 1.0F, 1.0F,
        -1.0F, -1.0F, 0.0F, 0.0F,
         1.0F,  1.0F, 1.0F, 1.0F,
        -1.0F,  1.0F, 0.0F, 1.0F,
    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

}  // namespace

GlCudaPresenter::GlCudaPresenter(const char* window_title)
    : window_title_(window_title)
{
}

GlCudaPresenter::~GlCudaPresenter()
{
    destroy_resources();

    if (window_ != nullptr)
    {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    if (glfw_initialized_)
    {
        glfwTerminate();
        glfw_initialized_ = false;
    }
}

void GlCudaPresenter::ensure_initialized(int width, int height)
{
    if (window_ != nullptr)
    {
        if (width != width_ || height != height_)
        {
            recreate_resources(width, height);
            glfwSetWindowSize(window_, width, height);
        }
        return;
    }

    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    glfw_initialized_ = true;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window_ = glfwCreateWindow(width, height, window_title_, nullptr, nullptr);
    if (window_ == nullptr)
    {
        throw std::runtime_error("Failed to create OpenGL window");
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
    {
        throw std::runtime_error("Failed to load OpenGL functions via glad");
    }

    shader_program_ = build_program();
    setup_fullscreen_quad(vao_, vbo_);
    recreate_resources(width, height);
}

void GlCudaPresenter::recreate_resources(int width, int height)
{
    if (cuda_pbo_resource_ != nullptr)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource_);
        cuda_pbo_resource_ = nullptr;
    }
    if (pbo_ != 0)
    {
        glDeleteBuffers(1, &pbo_);
        pbo_ = 0;
    }
    if (texture_ != 0)
    {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }

    width_ = width;
    height_ = height;

    glGenTextures(1, &texture_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    glGenBuffers(1, &pbo_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER,
                 static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_) * 4U,
                 nullptr,
                 GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    throw_cuda_if_failed(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource_, pbo_, cudaGraphicsMapFlagsWriteDiscard),
                         "cudaGraphicsGLRegisterBuffer");
}

void GlCudaPresenter::destroy_resources()
{
    if (cuda_pbo_resource_ != nullptr)
    {
        cudaGraphicsUnregisterResource(cuda_pbo_resource_);
        cuda_pbo_resource_ = nullptr;
    }

    if (pbo_ != 0)
    {
        glDeleteBuffers(1, &pbo_);
        pbo_ = 0;
    }

    if (texture_ != 0)
    {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }

    if (vbo_ != 0)
    {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }

    if (vao_ != 0)
    {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }

    if (shader_program_ != 0)
    {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }

    width_ = 0;
    height_ = 0;
}

void GlCudaPresenter::draw_frame()
{
    glViewport(0, 0, width_, height_);
    glClearColor(0.0F, 0.0F, 0.0F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shader_program_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glUniform1i(glGetUniformLocation(shader_program_, "image_texture"), 0);

    glBindVertexArray(vao_);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glfwSwapBuffers(window_);
}

void GlCudaPresenter::present(const tgpu::DeviceImageF32& image)
{
    if (image.empty())
    {
        return;
    }

    ensure_initialized(image.width, image.height);
    glfwMakeContextCurrent(window_);

    throw_cuda_if_failed(cudaGraphicsMapResources(1, &cuda_pbo_resource_, 0), "cudaGraphicsMapResources");

    unsigned char* mapped = nullptr;
    std::size_t mapped_size = 0;
    throw_cuda_if_failed(
        cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mapped), &mapped_size, cuda_pbo_resource_),
        "cudaGraphicsResourceGetMappedPointer");

    launch_f32_to_rgba_kernel(image.data, image.width, image.height, image.stride, mapped);
    throw_cuda_if_failed(cudaGetLastError(), "launch_f32_to_rgba_kernel");

    throw_cuda_if_failed(cudaGraphicsUnmapResources(1, &cuda_pbo_resource_, 0), "cudaGraphicsUnmapResources");

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    draw_frame();
}

void GlCudaPresenter::poll_events()
{
    if (window_ != nullptr)
    {
        glfwPollEvents();
    }
}

bool GlCudaPresenter::should_close() const
{
    if (window_ == nullptr)
    {
        return false;
    }

    const bool close_requested = glfwWindowShouldClose(window_) != 0;
    const bool esc_pressed = glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS;
    const bool q_pressed = glfwGetKey(window_, GLFW_KEY_Q) == GLFW_PRESS;
    return close_requested || esc_pressed || q_pressed;
}

}  // namespace tgpu::stream
