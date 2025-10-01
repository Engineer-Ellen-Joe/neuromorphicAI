#pragma once

#include "lve_window.hpp"
#include "lve_device.hpp"
#include "lve_renderer.hpp"
#include "lve_camera.hpp"
#include "debug/dear_imgui.hpp"
#include "systems/snn_data_reader.hpp"
#include "systems/neuron_render_system.hpp"

// std
#include <memory>
#include <vector>

namespace lve {
  class ControlApp {
  public:
    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;

    ControlApp();
    ~ControlApp();

    ControlApp(const ControlApp &) = delete;
    ControlApp &operator=(const ControlApp &) = delete;

    void run();

    void adjustZoom(float offset);

  private:
    LveWindow lveWindow{WIDTH, HEIGHT, "SNN Visualizer"};
    LveDevice lveDevice{lveWindow};
    LveRenderer lveRenderer{lveWindow, lveDevice};
    LveCamera camera{};

    std::unique_ptr<SnnDataReader> snnReader;
    std::unique_ptr<NeuronRenderSystem> neuronRenderSystem;
    std::unique_ptr<DearImgui> imgui_;
    std::vector<float> neuronStates;

    // Camera control members
    float zoom_ = 1.0f;
    glm::vec2 cameraPosition_ = {0.0f, 0.0f};
    glm::vec2 lastMousePos_ = {0.0f, 0.0f};
    bool isDragging_ = false;

    // Text visibility flags
    bool showInputText_ = false;
    bool showSynapseText_ = false;
    // bool showNeuronText_ = false;
  };
} // namespace lve