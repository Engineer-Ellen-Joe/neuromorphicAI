#include "control_app.hpp"

#include "lve_camera.hpp"

// libs
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

// std
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace lve {
  void ControlApp::adjustZoom(float offset) {
    zoom_ += offset * 0.1f;
    zoom_ = glm::clamp(zoom_, 0.1f, 10.0f);
  }

  // Callback function for mouse scroll
  void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
    auto app = reinterpret_cast<ControlApp *>(glfwGetWindowUserPointer(window));
    app->adjustZoom(static_cast<float>(yoffset));
  }

  ControlApp::ControlApp() {
    glfwSetWindowUserPointer(lveWindow.getGLFWwindow(), this);
    glfwSetScrollCallback(lveWindow.getGLFWwindow(), scroll_callback);

    const std::string SHM_NAME = "snn_visualization_shm";
    try {
      snnReader = std::make_unique<SnnDataReader>(SHM_NAME);
    } catch (const std::runtime_error &e) {
      std::cerr << "Error: " << e.what() << std::endl;
      throw;
    }

    neuronRenderSystem = std::make_unique<NeuronRenderSystem>(
        lveDevice, lveRenderer.getSwapChainRenderPass(), *snnReader);

    imgui_ = std::make_unique<DearImgui>(
        lveWindow, lveDevice, lveRenderer.getSwapChainRenderPass());
  }

  ControlApp::~ControlApp() {}

  void ControlApp::run() {
    while (!lveWindow.shouldClose()) {
      glfwPollEvents();

      if (snnReader) {
        snnReader->checkForNewData();
      }

      imgui_->newFrame();

      bool isMouseOverImGui = ImGui::GetIO().WantCaptureMouse;

      if (!isMouseOverImGui) {
        double xpos, ypos;
        glfwGetCursorPos(lveWindow.getGLFWwindow(), &xpos, &ypos);
        glm::vec2 currentMousePos = {static_cast<float>(xpos), static_cast<float>(ypos)};
        if (glfwGetMouseButton(lveWindow.getGLFWwindow(), GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
          if (isDragging_) {
            glm::vec2 delta = lastMousePos_ - currentMousePos;
            cameraPosition_ += delta * 0.002f * zoom_;
          }
          isDragging_ = true;
          lastMousePos_ = currentMousePos;
        } else {
          isDragging_ = false;
        }
      }

      float aspect = lveRenderer.getAspectRatio();
      camera.setOrthographicProjection(-aspect * zoom_, aspect * zoom_, -1.0f * zoom_, 1.0f * zoom_, -1.f, 1.f);
      camera.setViewDirection(glm::vec3(cameraPosition_, -1.0f), glm::vec3{0.f, 0.f, 1.f});

      ImGui::Begin("Debug Controls");
      ImGui::Checkbox("Show Synapses", &neuronRenderSystem->renderSynapses);
      ImGui::Checkbox("Show Input Values", &showInputText_);
      ImGui::Checkbox("Show Synapse Weights", &showSynapseText_);
      // ImGui::Checkbox("Show Neuron Competition", &showNeuronText_);
      ImGui::End();

      // Text Rendering Logic
      auto *drawList = ImGui::GetBackgroundDrawList();
      glm::mat4 viewProj = camera.getProjection() * camera.getView();
      auto windowSize = ImGui::GetIO().DisplaySize;

      /* if (showNeuronText_) {
          const auto& positions = neuronRenderSystem->neuronPositions;
          const auto& competition = snnReader->getCompetitionValues();
          if (positions.size() == competition.size()) { // Ensure data is consistent
              for (int i = 0; i < positions.size(); ++i) {
                  glm::vec4 worldPos = {positions[i].x, positions[i].y, 0, 1};
                  glm::vec4 clipPos = viewProj * worldPos;
                  glm::vec2 screenPos = {
                      (clipPos.x / clipPos.w + 1.0f) * 0.5f * windowSize.x,
                      (clipPos.y / clipPos.w + 1.0f) * 0.5f * windowSize.y}; // Corrected Y calculation
                  char text[16];
                  sprintf(text, "%.2f", competition[i]);
                  drawList->AddText(ImVec2(screenPos.x + 10, screenPos.y - 5), IM_COL32(255, 255, 0, 255), text);
              }
          }
      } */

      if (showInputText_) {
        const auto &targets = snnReader->getInputTargetIndices();
        const auto &weights = snnReader->getInputWeights();
        float inputStartX = -1.1f; // Should match the value in NeuronRenderSystem
        if (targets.size() == weights.size()) {
          for (int i = 0; i < targets.size(); ++i) {
            float startY = -0.9f + (1.8f * (i / (float)targets.size()));
            glm::vec4 worldPos = {inputStartX, startY, 0, 1};
            glm::vec4 clipPos = viewProj * worldPos;
            glm::vec2 screenPos = {
                (clipPos.x / clipPos.w + 1.0f) * 0.5f * windowSize.x,
                (clipPos.y / clipPos.w + 1.0f) * 0.5f * windowSize.y}; // Corrected Y calculation
            char text[16];
            sprintf(text, "%.2f", weights[i]);
            drawList->AddText(ImVec2(screenPos.x - 400, screenPos.y), IM_COL32(150, 150, 255, 255), text);
          }
        }
      }

      if (showSynapseText_) {
        const auto &sources = snnReader->getSourceIndices();
        const auto &targets = snnReader->getTargetIndices();
        const auto &weights = snnReader->getWeights();
        const auto &positions = neuronRenderSystem->neuronPositions;
        if (sources.size() == weights.size()) {
          for (int i = 0; i < sources.size(); ++i) {
            glm::vec2 p1 = positions[sources[i]];
            glm::vec2 p2 = positions[targets[i]];
            glm::vec2 midPoint = (p1 + p2) * 0.5f;
            glm::vec4 worldPos = {midPoint.x, midPoint.y, 0, 1};
            glm::vec4 clipPos = viewProj * worldPos;
            glm::vec2 screenPos = {
                (clipPos.x / clipPos.w + 1.0f) * 0.5f * windowSize.x,
                (clipPos.y / clipPos.w + 1.0f) * 0.5f * windowSize.y}; // Corrected Y calculation
            char text[16];
            sprintf(text, "%.2f", weights[i]);
            drawList->AddText(ImVec2(screenPos.x, screenPos.y), IM_COL32(200, 200, 200, 255), text);
          }
        }
      }

      if (auto commandBuffer = lveRenderer.beginFrame()) {
        int frameIndex = lveRenderer.getFrameindex();
        FrameInfo frameInfo{
            frameIndex, 0.f, commandBuffer, camera, VK_NULL_HANDLE, nullptr, nullptr};

        lveRenderer.beginSwapChainRenderPass(commandBuffer);
        neuronRenderSystem->render(frameInfo);
        imgui_->render(commandBuffer);
        lveRenderer.endSwapChainRenderPass(commandBuffer);
        lveRenderer.endFrame();
      }
    }
    vkDeviceWaitIdle(lveDevice.device());
  }

} // namespace lve
