cmake_minimum_required(VERSION 3.21)

project(pv162_tescan_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED CONFIG COMPONENTS imgcodecs imgproc)

add_library(tgpu_core
    src/common/image_io.cpp
    src/gpu/pipeline_gpu.cpp
)

target_include_directories(tgpu_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(tgpu_core
    PUBLIC
        opencv_imgcodecs
        opencv_imgproc
)

target_compile_features(tgpu_core PUBLIC cxx_std_20)

add_executable(tgpu_cli src/main.cpp)
target_link_libraries(tgpu_cli PRIVATE tgpu_core)
