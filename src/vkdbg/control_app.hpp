#pragma once

#include "lve_window.hpp"
#include "lve_device.hpp"
#include "lve_renderer.hpp"
#include "systems/snn_data_reader.hpp"

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

  private:
    LveWindow lveWindow{WIDTH, HEIGHT, "SNN Visualizer"};
    LveDevice lveDevice{lveWindow};
    LveRenderer lveRenderer{lveWindow, lveDevice};

    std::unique_ptr<SnnDataReader> snnReader;
    std::vector<float> neuronStates;
  };
} // namespace lve