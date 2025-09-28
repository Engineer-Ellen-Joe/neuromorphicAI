#include "simple_2d_render_system.hpp"

// std
#include <stdexcept>

namespace lve {

Simple2DRenderSystem::Simple2DRenderSystem(LveDevice& device, VkRenderPass renderPass)
    : lveDevice{device} {
  createPipelineLayout();
  createPipeline(renderPass);
}

Simple2DRenderSystem::~Simple2DRenderSystem() {
  vkDestroyPipelineLayout(lveDevice.device(), pipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(lveDevice.device(), descriptorSetLayout, nullptr);
}

void Simple2DRenderSystem::createPipelineLayout() {
  // Define a descriptor set layout for a single storage buffer
  VkDescriptorSetLayoutBinding layoutBinding{};
  layoutBinding.binding = 0;
  layoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  layoutBinding.descriptorCount = 1;
  layoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

  VkDescriptorSetLayoutCreateInfo layoutInfo{};
  layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layoutInfo.bindingCount = 1;
  layoutInfo.pBindings = &layoutBinding;

  if (vkCreateDescriptorSetLayout(lveDevice.device(), &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
      throw std::runtime_error("failed to create descriptor set layout!");
  }

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1; // We now have one layout
  pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pPushConstantRanges = nullptr;
  if (vkCreatePipelineLayout(lveDevice.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
    throw std::runtime_error("failed to create pipeline layout!");
  }
}

void Simple2DRenderSystem::createPipeline(VkRenderPass renderPass) {
  PipelineConfigInfo pipelineConfig{};
  LvePipeline::defaultPipelineConfigInfo(pipelineConfig);
  pipelineConfig.renderPass = renderPass;
  pipelineConfig.pipelineLayout = pipelineLayout;

  // Our new 2D shader doesn't use any vertex input
  pipelineConfig.bindingDescriptions.clear();
  pipelineConfig.attributeDescriptions.clear();

  lvePipeline = std::make_unique<LvePipeline>(
      lveDevice,
      "shaders/neuron.vert.spv",
      "shaders/neuron.frag.spv",
      pipelineConfig);
}

void Simple2DRenderSystem::render(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, uint32_t numNeurons) {
  lvePipeline->bind(commandBuffer);

  vkCmdBindDescriptorSets(
      commandBuffer,
      VK_PIPELINE_BIND_POINT_GRAPHICS,
      pipelineLayout,
      0, // first set
      1, // set count
      &descriptorSet,
      0, // dynamic offset count
      nullptr);

  // Draw 4 vertices per neuron instance
  vkCmdDraw(commandBuffer, numNeurons * 4, 1, 0, 0);
}

} // namespace lve
