cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.0/bin/nvcc")
project(add2 LANGUAGES CXX CUDA)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(/usr/include/python3.7)
include_directories(include)

set(SRCS kernel/add2.cpp kernel/add2.cu)

add_library(add2 SHARED ${SRCS})

target_link_libraries(add2 "${TORCH_LIBRARIES}")
