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

  ControlApp::ControlApp() {
    const std::string SHM_NAME = "snn_visualization_shm";
    const size_t NUM_NEURONS = 100;
    try {
        snnReader = std::make_unique<SnnDataReader>(SHM_NAME, NUM_NEURONS);
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        // Optionally, re-throw or handle the state to exit gracefully
        throw;
    }
    neuronStates.resize(NUM_NEURONS);
  }

  ControlApp::~ControlApp() {}

  void ControlApp::run() {
    auto currentTime = std::chrono::high_resolution_clock::now();

    while (!lveWindow.shouldClose()) {
      glfwPollEvents();

      if (snnReader && snnReader->checkForNewData()) {
          neuronStates = snnReader->getNeuronStates();
          // Print for verification, but not every single frame to avoid spam
          static int frameCount = 0;
          if (frameCount % 120 == 0) {
            std::cout << "[C++] New data received. Version: " << frameCount / 120 
                      << ", First neuron: " << neuronStates[0] << std::endl;
          }
          frameCount++;
      }

      auto newTime = std::chrono::high_resolution_clock::now();
      float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();
      currentTime = newTime;

      if (auto commandBuffer = lveRenderer.beginFrame()) {
        int frameIndex = lveRenderer.getFrameindex();
        
        lveRenderer.beginSwapChainRenderPass(commandBuffer);

        // render game objects

        lveRenderer.endSwapChainRenderPass(commandBuffer);
        lveRenderer.endFrame();
      }
    }

    vkDeviceWaitIdle(lveDevice.device());
  }
} // namespace lve