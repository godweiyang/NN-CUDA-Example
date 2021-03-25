cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
# modify to your own nvcc path, or delete it if ok
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(add2 LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Python REQUIRED)

if(NOT TF_CFLAGS)
  execute_process(
    COMMAND
      ${Python_EXECUTABLE} -c
      "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_CFLAGS
    RESULT_VARIABLE retcode)

  if(NOT "${retcode}" STREQUAL "0")
    message(
      FATAL_ERROR
        "Detecting TensorFlow Compile Flags - failed \n Did you installed TensorFlow?"
    )
  else()
    message(STATUS "Found TensorFlow Compile Flags: " ${TF_CFLAGS})
  endif()
endif()

if(NOT TF_LFLAGS)
  execute_process(
    COMMAND
      ${Python_EXECUTABLE} -c
      "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode)

  if(NOT "${retcode}" STREQUAL "0")
    message(
      FATAL_ERROR
        "Detecting TensorFlow Link Flags - failed \n Did you installed TensorFlow?"
    )
  else()
    message(STATUS "Found TensorFlow Link Flags: " ${TF_LFLAGS})
  endif()
endif()

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${TF_LFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TF_CFLAGS} -D GOOGLE_CUDA=1")

# modify to your own python path, or delete it if ok
include_directories(/usr/include/python3.7)
include_directories(../include)

set(SRCS add2_ops.cpp ../kernel/add2_kernel.cu)
add_library(add2 SHARED ${SRCS})
