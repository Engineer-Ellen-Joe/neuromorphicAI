#pragma once

#include "lve_camera.hpp"
#include "lve_descriptors.hpp"

// lib
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace lve{

    struct FrameInfo{
        int frameIndex;
        float frameTime;
        VkCommandBuffer commandBuffer;
        LveCamera &camera;
        VkDescriptorSet globalDescriptorSet;
        LveDescriptorPool &frameDescriptorPool;  // pool of descriptors that is cleared each frame
    };
} // namespace lve