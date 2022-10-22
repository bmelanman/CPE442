cmake_minimum_required(VERSION 3.23)
project(Sobel_Filter)

set(CMAKE_CXX_STANDARD 11)

include_directories(/opt/homebrew/opt/opencv/include/opencv4/)

add_executable(Sobel_Filter main.cpp)
