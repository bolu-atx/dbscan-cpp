cmake_minimum_required(VERSION 3.20)
project(dbscan-cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# SIMD optimization will be handled in the code with compiler intrinsics

# Library target
add_library(dbscan STATIC
    src/dbscan.cpp
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

# Tests
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

add_executable(dbscan_tests
    tests/test_dbscan.cpp
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

# Enable testing
enable_testing()
add_test(NAME dbscan_tests COMMAND dbscan_tests)