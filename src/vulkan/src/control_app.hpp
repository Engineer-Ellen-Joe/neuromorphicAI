#pragma once

#include "lve_window.hpp"
#include "lve_device.hpp"
#include "lve_renderer.hpp"
#include "lve_descriptors.hpp"
#include "systems/simple_2d_render_system.hpp"

// IPC and Threading
#include <zmq.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <map>

// std
#include <memory>

namespace lve {

  // 수신할 버퍼 정보를 담을 구조체
  struct GpuBufferInfo {
    std::vector<int> shape;
    std::string dtype;
    uint32_t size; // Total size of the buffer in bytes
    std::vector<char> ipc_handle; // Raw handle bytes
    std::string cuda_uuid;       // CUDA device UUID
    bool updated = false; // 새로 업데이트되었는지 여부
  };

  struct VulkanBuffer {
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDescriptorSet descriptorSet;
  };

  class ControlApp {
  public:
    static constexpr int WIDTH = 800;
    static constexpr int HEIGHT = 600;

    ControlApp();
    ~ControlApp();

    ControlApp(const LveWindow &) = delete;
    ControlApp &operator=(const LveWindow &) = delete;

    void run();

  private:
    void initZmq();
    void zmqReceiveLoop();

    LveWindow lveWindow{WIDTH, HEIGHT, "Ellen Project Debugger"};
    LveDevice lveDevice{lveWindow};
    LveRenderer lveRenderer{lveWindow, lveDevice};

    // Note: Descriptors might be simplified or removed later
    std::unique_ptr<LveDescriptorPool> globalPool{};
    std::unique_ptr<Simple2DRenderSystem> simple2DRenderSystem; 

    // ZMQ and Threading members
    std::unique_ptr<zmq::context_t> zmqContext;
    std::unique_ptr<zmq::socket_t> zmqSocket;
    std::thread zmqThread;
    std::atomic<bool> zmqThreadRunning{false};
    std::mutex dataMutex;
    std::map<std::string, GpuBufferInfo> gpuBuffers;
    std::map<std::string, VulkanBuffer> vulkanBuffers; // 임포트된 Vulkan 버퍼 저장
  };
} // namespace lve