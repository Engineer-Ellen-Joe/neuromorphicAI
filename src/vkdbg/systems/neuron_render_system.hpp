#pragma once

#include "lve_device.hpp"
#include "lve_pipeline.hpp"
#include "lve_frame_info.hpp"
#include "lve_buffer.hpp"
#include "snn_data_reader.hpp" // Include the data reader header

// std
#include <memory>
#include <vector>

namespace lve {

class NeuronRenderSystem {
public:
    // Constructor now accepts a reference to SnnDataReader
    NeuronRenderSystem(LveDevice& device, VkRenderPass renderPass, SnnDataReader& snnReader);
    ~NeuronRenderSystem();

    NeuronRenderSystem(const NeuronRenderSystem&) = delete;
    NeuronRenderSystem& operator=(const NeuronRenderSystem&) = delete;

    // render function no longer needs neuronStates parameter
    void render(FrameInfo& frameInfo);

    bool renderSynapses = true; // Public flag to control synapse rendering
    std::vector<glm::vec2> neuronPositions; // Expose neuron positions

private:
    void createPipelineLayout();
    void createNeuronPipeline(VkRenderPass renderPass);
    void createSynapsePipeline(VkRenderPass renderPass);

    void createNeuronVertexBuffer();
    void updateNeuronVertexBuffer();
    void createLineVertexBuffer();
    void updateSynapseVertexBuffer();

    LveDevice& lveDevice_;
    SnnDataReader& snnReader_; // Reference to the data reader

    // Pipelines
    std::unique_ptr<LvePipeline> neuronPipeline_;
    std::unique_ptr<LvePipeline> synapsePipeline_;
    VkPipelineLayout pipelineLayout_; // Can be shared for simple shaders

    // Vertex Buffers
    std::unique_ptr<LveBuffer> neuronVertexBuffer_;
    uint32_t neuronVertexCount_ = 0;
    std::unique_ptr<LveBuffer> synapseVertexBuffer_;
    uint32_t lineVertexCount_ = 0; // Total vertices for all lines
    uint32_t synapseVertexCount_ = 0; // Vertices for synapse lines only
};

} // namespace lve
