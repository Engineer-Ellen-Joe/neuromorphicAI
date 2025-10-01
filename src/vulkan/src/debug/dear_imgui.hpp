#pragma once

#include "lve_device.hpp"
#include "lve_window.hpp"

// IMGUI
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

// std
#include <stdexcept>

namespace lve {

    class DearImgui {
    public:
        DearImgui(LveWindow& window, LveDevice& device, VkRenderPass renderPass);
        ~DearImgui();

        void newFrame();
        void render(VkCommandBuffer commandBuffer);
    
    private:
        LveDevice& device_;
    };

} // namespace lve
