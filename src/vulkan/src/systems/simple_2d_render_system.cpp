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
}

void Simple2DRenderSystem::createPipelineLayout() {
  descriptorSetLayout = LveDescriptorSetLayout::Builder(lveDevice)
      .addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)
      .build();

  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  auto setLayout = descriptorSetLayout->getDescriptorSetLayout();
  pipelineLayoutInfo.pSetLayouts = &setLayout;
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
