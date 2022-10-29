cmake_minimum_required(VERSION 3.23)
project(Lab4)

# C++11 required for OpenCV
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_OSX_ARCHITECTURES "arm64")

add_compile_options(-pthread -O0)

# Get OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Declare the exe and link the OpenCV libs to it
add_executable(Lab4 main.cpp pthread_barrier.cpp pthread_barrier.h)

target_link_libraries(Lab4 ${OpenCV_LIBS})

set(CMAKE_ARGS "small_frog.mp4")

