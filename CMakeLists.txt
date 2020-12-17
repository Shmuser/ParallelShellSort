cmake_minimum_required(VERSION 3.17)
project(shell_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS -fopenmp)

add_executable(shell_test main.cpp)