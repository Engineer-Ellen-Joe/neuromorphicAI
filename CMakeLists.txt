cmake_minimum_required(VERSION 3.10)

project(EllenProject)

# Set the output directory for executables and libraries to the root of the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


# Add the vulkan engine library
add_subdirectory(src/vulkan)

# Add the vkdbg application
add_subdirectory(src/vkdbg)
