cmake_minimum_required(VERSION 3.21)

if(WIN32)
    set(_cuda_default_root "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1")
    if(NOT DEFINED ENV{CUDA_PATH} AND EXISTS "${_cuda_default_root}")
        set(ENV{CUDA_PATH} "${_cuda_default_root}")
    endif()
    if(NOT DEFINED ENV{CUDA_PATH_V13_1} AND EXISTS "${_cuda_default_root}")
        set(ENV{CUDA_PATH_V13_1} "${_cuda_default_root}")
    endif()
    if(EXISTS "$ENV{CUDA_PATH}")
        set(CMAKE_VS_GLOBALS "CudaToolkitDir=$ENV{CUDA_PATH}")
    endif()
endif()

project(pv162_tescan_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

include(CTest)
enable_testing()

find_package(CUDAToolkit REQUIRED)
find_package(Catch2 3 REQUIRED CONFIG)
include(Catch)
find_package(OpenCV REQUIRED CONFIG COMPONENTS imgcodecs imgproc)

add_library(tgpu_core
    src/common/image_io.cpp
    src/gpu/pipeline_gpu.cpp
    src/gpu/detail/pipeline.cu
    src/gpu/detail/memory.cu
    src/gpu/detail/input.cu
    src/gpu/detail/capture.cu
    src/gpu/stages/non_local_means.cu
    src/gpu/stages/unsharp_mask.cu
    src/gpu/stages/richardson_lucy.cu
    src/gpu/stages/histogram_stretch.cu
)

target_include_directories(tgpu_core
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src/gpu
)

target_link_libraries(tgpu_core
    PUBLIC
        opencv_imgcodecs
        opencv_imgproc
        CUDA::cudart
)

target_compile_features(tgpu_core PUBLIC cxx_std_20)

add_executable(tgpu_cli src/main.cpp)
target_link_libraries(tgpu_cli PRIVATE tgpu_core)

add_executable(tgpu_unit_tests
    tests/cpp/pipeline_tests.cpp
)

target_link_libraries(tgpu_unit_tests
    PRIVATE
        Catch2::Catch2WithMain
        tgpu_core
)

target_compile_features(tgpu_unit_tests PRIVATE cxx_std_20)

catch_discover_tests(tgpu_unit_tests)
