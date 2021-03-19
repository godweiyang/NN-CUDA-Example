cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(add2)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(/usr/include/python3.7)
include_directories(../include)

cuda_add_library(add2 SHARED add2.cpp add2.cu)

target_compile_features(add2 PRIVATE cxx_std_14)
target_link_libraries(add2 "${TORCH_LIBRARIES}")
