cmake_minimum_required(VERSION 3.11.0)

project(vkdbg CXX)

# Add the vulkan_engine library from the sibling directory


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

############## Build SHADERS for vkdbg #######################

find_program(GLSL_VALIDATOR glslangValidator HINTS
  ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}
  /usr/bin
  /usr/local/bin
  ${VULKAN_SDK_PATH}/Bin
  ${VULKAN_SDK_PATH}/Bin32
  $ENV{VULKAN_SDK}/Bin/
  $ENV{VULKAN_SDK}/Bin32/
)
if(NOT GLSL_VALIDATOR)
    message(FATAL_ERROR "glslangValidator not found.")
endif()

# Create a directory for shaders in the build output
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/shaders)

file(GLOB_RECURSE GLSL_SOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.frag"
  "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.vert"
)

foreach(GLSL ${GLSL_SOURCE_FILES})
  get_filename_component(FILE_NAME ${GLSL} NAME)
  set(SPIRV "${CMAKE_BINARY_DIR}/shaders/${FILE_NAME}.spv")
  add_custom_command(
    OUTPUT ${SPIRV}
    COMMAND ${GLSL_VALIDATOR} -V ${GLSL} -o ${SPIRV}
    DEPENDS ${GLSL})
  list(APPEND SPIRV_BINARY_FILES_VKDBG ${SPIRV})
endforeach(GLSL)

add_custom_target(
    Shaders_vkdbg
    DEPENDS ${SPIRV_BINARY_FILES_VKDBG}
)

add_dependencies(vkdbg Shaders_vkdbg)

