# FindTorch.cmake

# Reuse CMake's resolved interpreter so Torch discovery follows the configured
# environment instead of whichever `python` is first on PATH.
if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

# Try to determine the installation path of PyTorch using torch.utils.cmake prefix path
find_path(TORCH_CMAKE_PREFIX_PATH NAMES cmake/TorchConfig.cmake
  PATHS
    ENV TORCH_CMAKE_PREFIX_PATH
    DOC "Path to PyTorch cmake configuration."
)

# Use the configured Python interpreter to find it if the TORCH_CMAKE_PREFIX_PATH
# is not provided explicitly.
if(NOT TORCH_CMAKE_PREFIX_PATH)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_PREFIX_RESULT
  )
endif()

if(TORCH_PREFIX_RESULT AND NOT TORCH_CMAKE_PREFIX_PATH)
  message(FATAL_ERROR "Failed to query torch.utils.cmake_prefix_path with ${Python3_EXECUTABLE}")
endif()

message(STATUS "Found PyTorch CMake Prefix Path: ${TORCH_CMAKE_PREFIX_PATH}")

# Make sure the TORCH CMAKE PREFIX PATH points to the correct path
if(NOT IS_DIRECTORY ${TORCH_CMAKE_PREFIX_PATH})
  message(FATAL_ERROR "Invalid PyTorch cmake prefix path: ${TORCH_CMAKE_PREFIX_PATH}")
endif()

# Include PyTorch configuration
include(${TORCH_CMAKE_PREFIX_PATH}/ATen/ATenConfig.cmake)
include(${TORCH_CMAKE_PREFIX_PATH}/Torch/TorchConfig.cmake)
