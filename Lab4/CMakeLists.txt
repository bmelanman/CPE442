cmake_minimum_required(VERSION 3.23)
project(sobel_filter)

set(CMAKE_CXX_STANDARD 11)

include_directories(/opt/homebrew/opt/opencv/include/opencv4/)

add_executable(Lab4 main.cpp)
