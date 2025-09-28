#pragma once

#include "../lve_device.hpp"
#include "../lve_pipeline.hpp"

// std
#include <memory>

namespace lve {
  class Simple2DRenderSystem {
  public:
    Simple2DRenderSystem(LveDevice& device, VkRenderPass renderPass);
    ~Simple2DRenderSystem();

    Simple2DRenderSystem(const Simple2DRenderSystem &) = delete;
    Simple2DRenderSystem &operator=(const Simple2DRenderSystem &) = delete;

    void render(VkCommandBuffer commandBuffer, VkDescriptorSet descriptorSet, uint32_t numNeurons);

    VkDescriptorSetLayout const& getDescriptorSetLayout() const { return descriptorSetLayout; }

  private:
    void createPipelineLayout();
    void createPipeline(VkRenderPass renderPass);

    LveDevice& lveDevice;

    std::unique_ptr<LvePipeline> lvePipeline;
    VkPipelineLayout pipelineLayout;
    VkDescriptorSetLayout descriptorSetLayout; // 데이터 입력을 위한 레이아웃
  };
} // namespace lve
