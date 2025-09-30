cmake_minimum_required(VERSION 3.11.0)

project(vkdbg CXX)

# Add the vulkan_engine library from the sibling directory
add_subdirectory(../vulkan vulkan_engine_build)

# Gather all source files for the vkdbg executable
file(GLOB SOURCES "*.cpp")
file(GLOB SYSTEM_SOURCES "systems/*.cpp")
list(APPEND SOURCES ${SYSTEM_SOURCES})

# Create the executable
add_executable(vkdbg ${SOURCES})

# Link the executable against the vulkan_engine
# The engine handles linking its own dependencies like Vulkan, GLFW, etc.
target_link_libraries(vkdbg PRIVATE vulkan_engine)

# Set the C++ standard
target_compile_features(vkdbg PRIVATE cxx_std_17)

# Add the current directory to the include path
target_include_directories(vkdbg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set the working directory for debugging in Visual Studio
# This helps the executable find assets like shaders
set_property(TARGET vkdbg PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
