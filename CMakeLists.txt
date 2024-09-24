cmake_minimum_required(VERSION 3.25.2)

# Set compiler path
set(CUDAToolkit_ROOT /opt/cuda)
set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
set(CMAKE_CXX_COMPILER /usr/lib/ccache/g++-10)
set(CMAKE_C_COMPILER /usr/lib/ccache/gcc-10)

# Configure project's language standard
project(honeybadger_cpp LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_EXTENSIONS NO)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS NO)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Include FetchContent to fetch 3rd-party dependencies
include(FetchContent)

# Configure project
include(honeybadger/honeybadger.cmake)

# Enable OMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
