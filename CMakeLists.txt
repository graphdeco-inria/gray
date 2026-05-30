cmake_minimum_required(VERSION 3.24)
project(gray LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES native)

# * Set minimum requirements
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# * Use dynamic CUDA runtime instead of static
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# * Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
            "Choose the type of build, options are: Debug Release
RelWithDebInfo MinSizeRel."
            FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# * Add path to our CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

option(GRAY_BUILD_FAST_VIEWER "Build the standalone C++ fast viewer" OFF)

# * Dependency Discovery
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# We use this to be able to link against CUDA::cudart and/or CUDA::cuda_driver
find_package(CUDAToolkit REQUIRED)
find_package(OptiX8 REQUIRED)
find_package(Torch REQUIRED)

# * Source Files
set(SOURCES
    cuda/raytracer.cpp
    cuda/optix/bvh_wrapper.cu
    cuda/utils/sh.cu
    cuda/opt/adam.cu
)

# * Main Shared Library Target
add_library(gray SHARED ${SOURCES})

# * OptiX PTX Target
add_library(optix_program OBJECT "cuda/shaders.cu")
set_target_properties(optix_program PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_PTX_COMPILATION ON
)

# * Include Directories
target_include_directories(gray PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OptiX8_INCLUDE_DIRS}
)

target_include_directories(optix_program PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${OptiX8_INCLUDE_DIRS}
)

# * Compile-time configuration
set(CHANNELS "3" CACHE STRING "Number of gaussian channels")
set(MAX_ALPHA "0.9999" CACHE STRING "Maximum alpha value for opacity")
target_compile_definitions(gray PRIVATE CHANNELS=${CHANNELS} MAX_ALPHA=${MAX_ALPHA})
target_compile_definitions(optix_program PRIVATE CHANNELS=${CHANNELS} MAX_ALPHA=${MAX_ALPHA})

# * Definitions
target_compile_definitions(gray PRIVATE
    CONFIG=${CONFIG}
    ENABLE_OPTIX
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    CMAKE_CURRENT_LIST_DIR="${CMAKE_CURRENT_LIST_DIR}"
    OPTIX_SAMPLE_NAME_DEFINE=gray
    OPTIX_SAMPLE_DIR_DEFINE=gray
)
target_compile_definitions(optix_program PRIVATE
    CONFIG=${CONFIG}
)

# * Compile Options
target_compile_options(gray PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ftz=true --prec-div=false --prec-sqrt=false -Xptxas=-O3,-dlcm=ca>
)
if(NOT WIN32)
    target_compile_options(gray PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -fno-math-errno -fno-trapping-math>
    )
endif()
target_compile_options(optix_program PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ftz=true --prec-div=false --prec-sqrt=false>
)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)

# * Link Libraries
target_link_libraries(gray PRIVATE
    Python3::Python
    ${TORCH_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    OptiX8
)
target_link_libraries(optix_program PRIVATE
    ${TORCH_LIBRARIES}
    OptiX8
)

# * Custom: copy the .ptx next to the .so
set(PTX_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/optix_program.dir/cuda/shaders.ptx")
set(PTX_DEST "${CMAKE_CURRENT_BINARY_DIR}/libgray.ptx")
add_custom_command(
    OUTPUT ${PTX_DEST}
    DEPENDS ${PTX_SOURCE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${PTX_SOURCE}" "${PTX_DEST}"
    COMMENT "Copying PTX file (libgray.ptx) to build output directory"
)
add_custom_target(copy_optix_ptx ALL DEPENDS ${PTX_DEST})
add_dependencies(copy_optix_ptx optix_program)

include(cmake/fast_viewer.cmake)

install(FILES ${PTX_DEST} DESTINATION gray)

# * For pip install: put the .so in the resulting module
install(TARGETS gray DESTINATION gray)

# * Ensure MSVC uses /MD
if(MSVC)
  set_property(TARGET gray PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()
