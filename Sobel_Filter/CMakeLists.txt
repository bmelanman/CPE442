cmake_minimum_required(VERSION 3.23)
project(Sobel_Filter)

# C++11 required for OpenCV
set(CMAKE_CXX_STANDARD 11)

add_compile_options(-O3)

# Get OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Declare the exe and link the OpenCV libs to it

if (APPLE)
    add_executable(Sobel_Filter src/main.cpp lib/pthread_barrier.cpp lib/pthread_barrier.h)
else ()
    add_executable(Sobel_Filter src/main.cpp)
endif ()

target_link_libraries(Sobel_Filter ${OpenCV_LIBS})

set(CMAKE_ARGS "small_frog.mp4")

