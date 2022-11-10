cmake_minimum_required(VERSION 3.23)
project(Sobel_Tests)

set(CMAKE_CXX_STANDARD 11)

# C++11 required for OpenCV
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_OSX_ARCHITECTURES "arm64")

add_compile_options(-Wall -pthread -O0)

# Get OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(brick main.cpp)
add_executable(gavin main2.cpp)
add_executable(jonch josh5.cpp lib/pthread_barrier.cpp lib/pthread_barrier.h)

# Declare the target and link the OpenCV libs to it
target_link_libraries(brick ${OpenCV_LIBS})
target_link_libraries(gavin ${OpenCV_LIBS})
target_link_libraries(jonch ${OpenCV_LIBS})

