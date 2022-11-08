cmake_minimum_required(VERSION 3.23)
project(Lab6)

# C++11 required for OpenCV
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OSX_ARCHITECTURES "arm64")

add_compile_options(-Wall -pthread -O0)

# Get OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

if (APPLE)
    add_executable(Lab6 src/main.cpp lib/pthread_barrier.cpp lib/pthread_barrier.h)
    add_compile_options(-Wall -mavx -mfma)
else ()
    add_executable(Lab6 src/main.cpp)
endif ()

# Declare the target and link the OpenCV libs to it
target_link_libraries(Lab6 ${OpenCV_LIBS})

set(CMAKE_ARGS "small_frog.mp4")
