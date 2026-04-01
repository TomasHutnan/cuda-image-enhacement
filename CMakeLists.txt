cmake_minimum_required(VERSION 3.21)

project(pv162_tescan_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Allow local/preset overrides while keeping a sensible default for lab machines.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

include(CTest)
enable_testing()

find_package(CUDAToolkit REQUIRED)
find_package(Catch2 3 REQUIRED CONFIG)
include(Catch)
find_package(OpenCV REQUIRED CONFIG COMPONENTS imgcodecs imgproc highgui)

set(TGPU_PUBLIC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(TGPU_PRIVATE_GPU_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src/gpu)
set(TGPU_PRIVATE_SRC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/src)

add_library(tgpu_core
    src/common/image_io.cpp
    src/gpu/pipeline_gpu.cpp
    src/gpu/detail/pipeline.cu
    src/gpu/detail/buffer_io.cu
    src/gpu/stages/non_local_means.cu
    src/gpu/stages/unsharp_mask.cu
    src/gpu/stages/richardson_lucy.cu
    src/gpu/stages/histogram_stretch.cu
)

target_include_directories(tgpu_core
    PUBLIC
        ${TGPU_PUBLIC_INCLUDE_DIR}
    PRIVATE
        ${TGPU_PRIVATE_GPU_INCLUDE_DIR}
)

target_link_libraries(tgpu_core
    PUBLIC
        opencv_imgcodecs
        opencv_imgproc
        CUDA::cudart
)

target_compile_features(tgpu_core PUBLIC cxx_std_20)

function(add_tgpu_app target_name)
    add_executable(${target_name} ${ARGN})
    target_include_directories(${target_name} PRIVATE ${TGPU_PRIVATE_SRC_INCLUDE_DIR})
    target_link_libraries(${target_name} PRIVATE tgpu_core)
endfunction()

add_tgpu_app(tgpu_cli
    src/main.cpp
    src/cli/cli_options.cpp
    src/cli/reporting.cpp
    src/cli/processing.cpp
)

add_tgpu_app(tgpu_stream_viewer
    src/stream_viewer.cpp
)
target_link_libraries(tgpu_stream_viewer PRIVATE tgpu_core opencv_highgui)

add_executable(tgpu_unit_tests
    tests/cpp/core/image_tests.cpp
    tests/cpp/core/pipeline_tests.cpp
    tests/cpp/stages/nlm_stage_tests.cpp
    tests/cpp/stages/unsharp_mask_stage_tests.cpp
    tests/cpp/stages/richardson_lucy_stage_tests.cpp
    tests/cpp/stages/histogram_stretch_stage_tests.cpp
)

target_link_libraries(tgpu_unit_tests
    PRIVATE
        Catch2::Catch2WithMain
        tgpu_core
)

target_compile_features(tgpu_unit_tests PRIVATE cxx_std_20)

catch_discover_tests(tgpu_unit_tests)
