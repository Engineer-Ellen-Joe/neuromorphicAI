#pragma once

#include "lve_device.hpp"
#include "lve_frame_info.hpp"

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

namespace lve{
  class Dimgui{
  public:
    void init(LveDevice &device, VkRenderPass renderPass, uint32_t imageCount, GLFWwindow *window);

    void newFrame();
    void update(const FrameInfo &frameInfo); // ← 시스템처럼 처리
    void render(VkCommandBuffer commandBuffer);
    void cleanup();

  private:
    VkDevice device{};
    VkDescriptorPool descriptorPool{};
    bool initialized = false;
  };
} // namespace lve