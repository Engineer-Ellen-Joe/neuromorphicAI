#include "neuron_render_system.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

// std
#include <array>
#include <cassert>
#include <stdexcept>
#include <iostream>

namespace lve {

// Simple vertex structure, used for both neurons and synapses
struct Vertex {
    glm::vec2 position;
    glm::vec3 color;
    glm::vec2 local_pos; // Position relative to the center of the quad (-1 to 1)


    static std::vector<VkVertexInputBindingDescription> getBindingDescriptions();
    static std::vector<VkVertexInputAttributeDescription> getAttributeDescriptions();
};

std::vector<VkVertexInputBindingDescription> Vertex::getBindingDescriptions() {
    std::vector<VkVertexInputBindingDescription> bindingDescriptions(1);
    bindingDescriptions[0].binding = 0;
    bindingDescriptions[0].stride = sizeof(Vertex);
    bindingDescriptions[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    return bindingDescriptions;
}

std::vector<VkVertexInputAttributeDescription> Vertex::getAttributeDescriptions() {
    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Vertex, position);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Vertex, color);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Vertex, local_pos);
    return attributeDescriptions;
}

NeuronRenderSystem::NeuronRenderSystem(LveDevice& device, VkRenderPass renderPass, SnnDataReader& snnReader)
    : lveDevice_{device}, snnReader_{snnReader} {
    createPipelineLayout();
    createNeuronPipeline(renderPass);
    createSynapsePipeline(renderPass);
}

NeuronRenderSystem::~NeuronRenderSystem() {
    vkDestroyPipelineLayout(lveDevice_.device(), pipelineLayout_, nullptr);
}

void NeuronRenderSystem::createPipelineLayout() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4) * 2; // For proj and view matrices

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 0;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    if (vkCreatePipelineLayout(lveDevice_.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout_) !=
        VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }
}

void NeuronRenderSystem::createNeuronPipeline(VkRenderPass renderPass) {
    assert(pipelineLayout_ != nullptr && "Cannot create pipeline before pipeline layout");
    PipelineConfigInfo pipelineConfig{};
    LvePipeline::defaultPipelineConfigInfo(pipelineConfig);
    pipelineConfig.renderPass = renderPass;
    pipelineConfig.pipelineLayout = pipelineLayout_;
    pipelineConfig.bindingDescriptions = Vertex::getBindingDescriptions();
    pipelineConfig.attributeDescriptions = Vertex::getAttributeDescriptions();

    neuronPipeline_ = std::make_unique<LvePipeline>(
        lveDevice_,
        "build/shaders/simple_2d.vert.spv",
        "build/shaders/simple_2d.frag.spv",
        pipelineConfig);
}

void NeuronRenderSystem::createSynapsePipeline(VkRenderPass renderPass) {
    assert(pipelineLayout_ != nullptr && "Cannot create pipeline before pipeline layout");
    PipelineConfigInfo pipelineConfig{};
    LvePipeline::defaultPipelineConfigInfo(pipelineConfig);
    pipelineConfig.renderPass = renderPass;
    pipelineConfig.pipelineLayout = pipelineLayout_;
    // Key difference: draw lines instead of triangles
    pipelineConfig.inputAssemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    pipelineConfig.bindingDescriptions = Vertex::getBindingDescriptions();
    pipelineConfig.attributeDescriptions = Vertex::getAttributeDescriptions();

    synapsePipeline_ = std::make_unique<LvePipeline>(
        lveDevice_,
        "build/shaders/simple_2d.vert.spv",
        "build/shaders/simple_2d.frag.spv",
        pipelineConfig);
}

void NeuronRenderSystem::createNeuronVertexBuffer() {
    const int totalNeurons = snnReader_.getTotalNeurons();
    if (totalNeurons == 0) return;

    neuronPositions.resize(totalNeurons);
    const int verticesPerNeuron = 6;
    neuronVertexCount_ = totalNeurons * verticesPerNeuron;
    std::vector<Vertex> vertices(neuronVertexCount_);

    const auto& layers = snnReader_.getNeuronsPerLayer();
    const int numLayers = snnReader_.getNumLayers();
    float layerSpacing = (numLayers > 1) ? 1.8f / (numLayers - 1) : 0;
    float startX = (numLayers > 1) ? -0.9f : 0.0f;

    int neuronIdxOffset = 0;
    for (int i = 0; i < numLayers; ++i) {
        int neuronsInLayer = layers[i];
        float neuronSpacing = (neuronsInLayer > 1) ? 1.8f / (neuronsInLayer - 1) : 0;
        float startY = (neuronsInLayer > 1) ? 0.9f : 0.0f;
        float layerX = startX + i * layerSpacing;

        for (int j = 0; j < neuronsInLayer; ++j) {
            int currentNeuronIndex = neuronIdxOffset + j;
            float neuronY = startY - j * neuronSpacing;
            neuronPositions[currentNeuronIndex] = {layerX, neuronY};

            float size = 0.05f; 

            // Define local positions for the quad corners
            glm::vec2 tl_local = {-1.0f, 1.0f};
            glm::vec2 tr_local = {1.0f, 1.0f};
            glm::vec2 bl_local = {-1.0f, -1.0f};
            glm::vec2 br_local = {1.0f, -1.0f};

            vertices[currentNeuronIndex * verticesPerNeuron + 0] = {{layerX - size, neuronY + size}, {0.5f, 0.5f, 0.5f}, tl_local};
            vertices[currentNeuronIndex * verticesPerNeuron + 1] = {{layerX - size, neuronY - size}, {0.5f, 0.5f, 0.5f}, bl_local};
            vertices[currentNeuronIndex * verticesPerNeuron + 2] = {{layerX + size, neuronY + size}, {0.5f, 0.5f, 0.5f}, tr_local};
            vertices[currentNeuronIndex * verticesPerNeuron + 3] = {{layerX + size, neuronY + size}, {0.5f, 0.5f, 0.5f}, tr_local};
            vertices[currentNeuronIndex * verticesPerNeuron + 4] = {{layerX - size, neuronY - size}, {0.5f, 0.5f, 0.5f}, bl_local};
            vertices[currentNeuronIndex * verticesPerNeuron + 5] = {{layerX + size, neuronY - size}, {0.5f, 0.5f, 0.5f}, br_local};
        }
        neuronIdxOffset += neuronsInLayer;
    }

    uint32_t vertexSize = sizeof(vertices[0]);
    neuronVertexBuffer_ = std::make_unique<LveBuffer>(
        lveDevice_, vertexSize, neuronVertexCount_,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    neuronVertexBuffer_->map();
    neuronVertexBuffer_->writeToBuffer((void*)vertices.data());
}

void NeuronRenderSystem::updateNeuronVertexBuffer() {
    if (!neuronVertexBuffer_) return;
    auto* vertices = static_cast<Vertex*>(neuronVertexBuffer_->getMappedMemory());
    if (!vertices) return;

    const auto& neuronStates = snnReader_.getNeuronStates();
    if (neuronStates.empty()) return;

    for (int i = 0; i < neuronStates.size(); ++i) {
        float activation = neuronStates[i];
        glm::vec3 color = (activation > 0.5f) ? glm::vec3(1.0f, 1.0f, 0.0f) : glm::vec3(0.2f, 0.2f, 0.2f);
        for (int j = 0; j < 6; ++j) {
            vertices[i * 6 + j].color = color;
        }
    }
}

void NeuronRenderSystem::createLineVertexBuffer() {
    const int numConnections = snnReader_.getNumConnections();
    const int numInputSynapses = snnReader_.getNumInputSynapses();
    const int numLayers = snnReader_.getNumLayers();
    if (numConnections == 0 && numInputSynapses == 0) return;

    std::vector<Vertex> vertices;
    vertices.reserve((numConnections + numInputSynapses + numLayers * 4) * 2);

    // 1. Inter-neuron synapses
    if (numConnections > 0) {
        const auto& sourceIndices = snnReader_.getSourceIndices();
        const auto& targetIndices = snnReader_.getTargetIndices();
        const auto& weights = snnReader_.getWeights();
        for (int i = 0; i < numConnections; ++i) {
            int sourceIdx = sourceIndices[i];
            int targetIdx = targetIndices[i];
            if (sourceIdx >= neuronPositions.size() || targetIdx >= neuronPositions.size()) continue;
            glm::vec3 color = {weights[i], weights[i], weights[i] * 0.5f};
            vertices.push_back({neuronPositions[sourceIdx], color});
            vertices.push_back({neuronPositions[targetIdx], color});
        }
    }

    // 2. Input synapses
    if (numInputSynapses > 0) {
        const auto& inputTargetIndices = snnReader_.getInputTargetIndices();
        float inputStartX = -2.0f; // 입력 데이터의 시작 위치를 지정
        for (int i = 0; i < numInputSynapses; ++i) {
            int targetIdx = inputTargetIndices[i];
            if (targetIdx >= neuronPositions.size()) continue;
            float startY = -0.9f + (1.8f * (i / (float)numInputSynapses)); 
            glm::vec3 color = {0.5f, 0.5f, 1.0f};
            vertices.push_back({{inputStartX, startY}, color});
            vertices.push_back({neuronPositions[targetIdx], color});
        }
    }

    synapseVertexCount_ = static_cast<uint32_t>(vertices.size());

    // 3. Layer bounding boxes
    int neuronIdxOffset = 0;
    const auto& layers = snnReader_.getNeuronsPerLayer();
    for (int i = 0; i < numLayers; ++i) {
        if (layers[i] == 0) continue;
        glm::vec2 firstNeuronPos = neuronPositions[neuronIdxOffset];
        glm::vec2 lastNeuronPos = neuronPositions[neuronIdxOffset + layers[i] - 1];
        float boxWidth = 0.15f;
        glm::vec2 topLeft = {firstNeuronPos.x - boxWidth, firstNeuronPos.y + boxWidth};
        glm::vec2 bottomRight = {lastNeuronPos.x + boxWidth, lastNeuronPos.y - boxWidth};
        glm::vec3 boxColor = {0.8f, 0.8f, 0.8f}; // White box

        vertices.push_back({topLeft, boxColor}); // top left -> top right
        vertices.push_back({{bottomRight.x, topLeft.y}, boxColor});
        vertices.push_back({{bottomRight.x, topLeft.y}, boxColor}); // top right -> bottom right
        vertices.push_back({bottomRight, boxColor});
        vertices.push_back({bottomRight, boxColor}); // bottom right -> bottom left
        vertices.push_back({{topLeft.x, bottomRight.y}, boxColor});
        vertices.push_back({{topLeft.x, bottomRight.y}, boxColor}); // bottom left -> top left
        vertices.push_back({topLeft, boxColor});

        neuronIdxOffset += layers[i];
    }

    lineVertexCount_ = static_cast<uint32_t>(vertices.size());
    if (lineVertexCount_ == 0) return;

    uint32_t vertexSize = sizeof(vertices[0]);
    synapseVertexBuffer_ = std::make_unique<LveBuffer>(
        lveDevice_, vertexSize, lineVertexCount_,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    synapseVertexBuffer_->map();
    synapseVertexBuffer_->writeToBuffer((void*)vertices.data());
}

void NeuronRenderSystem::updateSynapseVertexBuffer() {
    if (!synapseVertexBuffer_) return;
    auto* vertices = static_cast<Vertex*>(synapseVertexBuffer_->getMappedMemory());
    if (!vertices) return;

    const auto& weights = snnReader_.getWeights();
    if (weights.empty()) return;

    for (int i = 0; i < weights.size(); ++i) {
        float weight = weights[i];
        glm::vec3 color = {weight, weight, weight * 0.5f}; // yellowish tint for higher weights
        vertices[i * 2 + 0].color = color;
        vertices[i * 2 + 1].color = color;
    }
}

void NeuronRenderSystem::render(FrameInfo& frameInfo) {
    if (snnReader_.getTotalNeurons() == 0) return;

    if (!neuronVertexBuffer_) createNeuronVertexBuffer();
    if (!synapseVertexBuffer_ && (snnReader_.getNumConnections() > 0 || snnReader_.getNumInputSynapses() > 0)) createLineVertexBuffer();

    updateNeuronVertexBuffer();
    updateSynapseVertexBuffer();

    // Push camera matrices
    struct PushConstantData {
        glm::mat4 proj;
        glm::mat4 view;
    } pushData;
    pushData.proj = frameInfo.camera.getProjection();
    pushData.view = frameInfo.camera.getView();

    vkCmdPushConstants(
        frameInfo.commandBuffer,
        pipelineLayout_,
        VK_SHADER_STAGE_VERTEX_BIT,
        0,
        sizeof(PushConstantData),
        &pushData);

    // Draw lines (synapses and boxes)
    if (synapseVertexBuffer_) {
        synapsePipeline_->bind(frameInfo.commandBuffer);
        VkBuffer synapseBuffers[] = {synapseVertexBuffer_->getBuffer()};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(frameInfo.commandBuffer, 0, 1, synapseBuffers, offsets);

        // Draw synapses conditionally
        if (renderSynapses && synapseVertexCount_ > 0) {
            vkCmdDraw(frameInfo.commandBuffer, synapseVertexCount_, 1, 0, 0);
        }

        // Always draw layer boxes
        uint32_t boxVertexCount = lineVertexCount_ - synapseVertexCount_;
        if (boxVertexCount > 0) {
            vkCmdDraw(frameInfo.commandBuffer, boxVertexCount, 1, synapseVertexCount_, 0);
        }
    }

    // Draw neurons on top (foreground)
    if (neuronVertexBuffer_) {
        neuronPipeline_->bind(frameInfo.commandBuffer);
        VkBuffer neuronBuffers[] = {neuronVertexBuffer_->getBuffer()};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(frameInfo.commandBuffer, 0, 1, neuronBuffers, offsets);
        vkCmdDraw(frameInfo.commandBuffer, neuronVertexCount_, 1, 0, 0);
    }
}

} // namespace lve
