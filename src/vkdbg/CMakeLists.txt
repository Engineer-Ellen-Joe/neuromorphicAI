cmake_minimum_required(VERSION 3.18)
project(vkdbg)

# Add this line to define the macro for all targets
add_compile_definitions(VK_USE_PLATFORM_WIN32_KHR)

find_package(pybind11 CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# Add include directories with absolute path
include_directories("C:/Dev/2025/EllenProject/src/vulkan/src")

# --- Add Lve source files with absolute path ---
set(LVE_SOURCES
    "C:/Dev/2025/EllenProject/src/vulkan/src/lve_window.cpp"
    "C:/Dev/2025/EllenProject/src/vulkan/src/lve_device.cpp"
)

# --- vkdbg module ---
pybind11_add_module(vkdbg vkdbg.cpp)
target_link_libraries(vkdbg PRIVATE Vulkan::Vulkan)
set_target_properties(vkdbg PROPERTIES LIBRARY_OUTPUT_DIRECTORY "C:/Dev/2025/EllenProject/python")

# --- vulkan_setup module ---
pybind11_add_module(vulkan_setup vulkan_setup.cpp ${LVE_SOURCES})
target_link_libraries(vulkan_setup PRIVATE Vulkan::Vulkan glfw)
set_target_properties(vulkan_setup PROPERTIES LIBRARY_OUTPUT_DIRECTORY "C:/Dev/2025/EllenProject/python")