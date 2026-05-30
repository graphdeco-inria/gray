if(NOT GRAY_BUILD_FAST_VIEWER)
    return()
endif()

if(WIN32)
    find_package(OpenGL REQUIRED)
    find_package(glfw3 CONFIG REQUIRED)
else()
    find_package(OpenGL REQUIRED)
    find_package(X11 REQUIRED)
endif()

add_executable(fast_viewer
    cuda/fast_viewer/fast_viewer.cpp
    cuda/fast_viewer/fast_viewer_kernels.cu
    cuda/optix/bvh_wrapper.cu
    cuda/utils/sh.cu
    cuda/opt/adam.cu
)

target_include_directories(fast_viewer PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${OptiX8_INCLUDE_DIRS}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${OPENGL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/cuda
)

if(NOT WIN32)
    target_include_directories(fast_viewer PRIVATE
        ${X11_INCLUDE_DIR}
    )
endif()

target_compile_definitions(fast_viewer PRIVATE
    CHANNELS=${CHANNELS}
    MAX_ALPHA=${MAX_ALPHA}
    CONFIG=${CONFIG}
    ENABLE_OPTIX
    CMAKE_BINARY_DIR="${CMAKE_BINARY_DIR}"
    CMAKE_CURRENT_LIST_DIR="${CMAKE_SOURCE_DIR}"
    $<$<BOOL:$ENV{EXTRA_H_AVAILABLE}>:EXTRA_H_AVAILABLE=1>
    OPTIX_SAMPLE_NAME_DEFINE=gray
    OPTIX_SAMPLE_DIR_DEFINE=gray
)

target_compile_options(fast_viewer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --ftz=true --prec-div=false --prec-sqrt=false -Xptxas=-O3,-dlcm=ca>
)

if(NOT WIN32)
    target_compile_options(fast_viewer PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -DNDEBUG -march=native -fno-math-errno -fno-trapping-math>
    )
endif()

target_link_libraries(fast_viewer PRIVATE
    Python3::Python
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    ${OptiX_LIBRARIES}
    OpenGL::GL
)

if(WIN32)
    target_link_libraries(fast_viewer PRIVATE
        glfw
        opengl32
    )
else()
    target_link_libraries(fast_viewer PRIVATE
        ${X11_LIBRARIES}
        dl
    )
endif()

add_dependencies(fast_viewer copy_optix_ptx)

if(MSVC)
    set_property(TARGET fast_viewer PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
endif()