cmake_minimum_required(VERSION 3.20)
project(dbscan-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Use native architecture on macOS (allows universal builds if supported)
if(APPLE)
    # Let CMake detect the native architecture
    # This allows building for both x86_64 and arm64 when possible
    message(STATUS "macOS detected - using native architecture: ${CMAKE_OSX_ARCHITECTURES}")
endif()

# SIMD optimization will be handled in the code with compiler intrinsics

# Library target
add_library(dbscan STATIC
    src/dbscan.cpp
    src/dbscan_optimized.cpp
    src/dbscan_grid2d_l1.cpp
)

target_include_directories(dbscan
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        src
        include
)

target_compile_options(dbscan PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -O3
)

target_link_libraries(dbscan
    PRIVATE
)

# Dependencies
include(FetchContent)

# Catch2 for testing
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# nanobench for benchmarking
FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
)
FetchContent_MakeAvailable(nanobench)


add_executable(dbscan_tests
    tests/test_dbscan.cpp
    tests/test_dbscan_optimized.cpp
    tests/test_dbscan_grid2d_l1.cpp
    tests/test_parallel_for.cpp
    tests/test_union_find.cpp
)

target_link_libraries(dbscan_tests
    PRIVATE
        dbscan
        Catch2::Catch2WithMain
)

target_include_directories(dbscan_tests
    PRIVATE
        include
)

add_executable(dbscan_dataset_validator
    tools/dbscan_dataset_validator.cpp
)

target_link_libraries(dbscan_dataset_validator
    PRIVATE
        dbscan
)

target_include_directories(dbscan_dataset_validator
    PRIVATE
        include
)

# Benchmark executable
add_executable(dbscan_benchmark
    benchmark/benchmark_dbscan.cpp
)

target_link_libraries(dbscan_benchmark
    PRIVATE
        dbscan
        nanobench
)

# Note: ThreadPool is not currently used in the optimized implementation
# It was designed for parallel processing but the current implementation
# uses sequential processing for simplicity and correctness

target_include_directories(dbscan_benchmark
    PRIVATE
        include
)

# Enable testing
enable_testing()
add_test(NAME dbscan_tests COMMAND dbscan_tests)

add_test(
    NAME dbscan_dataset_validation
    COMMAND ${CMAKE_COMMAND}
        -DPROJECT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DPROJECT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/run_dataset_validator.cmake
)
