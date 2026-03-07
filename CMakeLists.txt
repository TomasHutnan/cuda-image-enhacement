cmake_minimum_required(VERSION 3.21)

project(pv162_tescan_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

find_package(Catch2 3 REQUIRED CONFIG)
include(Catch)
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
