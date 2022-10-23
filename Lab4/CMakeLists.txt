cmake_minimum_required(VERSION 3.23)
project(Lab4)

# C++11 required for OpenCV
set(CMAKE_CXX_STANDARD 11)

# Get OpenCV libs
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Declare the exe and link the OpenCV libs to it
add_executable(Lab4 main.cpp)
target_link_libraries(Lab4 ${OpenCV_LIBS})

