#include "control_app.hpp"

// std
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>               // For std::stringstream and std::hex, std::setw, std::setfill
#include <iomanip>               // For std::setw, std::setfill
#include <nlohmann/json.hpp>     // For parsing metadata
#include <windows.h>             // For HANDLE
#include <vulkan/vulkan_win32.h> // For VkImportMemoryWin32HandleInfoKHR

namespace lve{

  ControlApp::ControlApp() {
    // Descriptor pool might be needed for future 2D rendering shaders
    globalPool = LveDescriptorPool::Builder(lveDevice)
                     .setMaxSets(LveSwapChain::MAX_FRAMES_IN_FLIGHT)
                     .addPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, LveSwapChain::MAX_FRAMES_IN_FLIGHT)
                     .build();

    simple2DRenderSystem = std::make_unique<Simple2DRenderSystem>(lveDevice, lveRenderer.getSwapChainRenderPass());

    initZmq();
  }

  ControlApp::~ControlApp() {
    zmqThreadRunning = false;
    if (zmqThread.joinable()) { zmqThread.join(); }

    // Destroy all imported Vulkan buffers and their memory
    for (auto const &[name, buffer] : vulkanBuffers) {
      vkDestroyBuffer(lveDevice.device(), buffer.buffer, nullptr);
      vkFreeMemory(lveDevice.device(), buffer.memory, nullptr);
    }
  }

  void ControlApp::initZmq() {
    zmqContext = std::make_unique<zmq::context_t>(1);
    zmqSocket = std::make_unique<zmq::socket_t>(*zmqContext, zmq::socket_type::sub);
    try {
      zmqSocket->connect("tcp://localhost:5555");
      zmqSocket->set(zmq::sockopt::subscribe, ""); // Subscribe to all topics
      std::cout << "[ZMQ] Subscriber connected to tcp://localhost:5555" << std::endl;
    }
    catch (const zmq::error_t &e) {
      std::cerr << "[ZMQ] Connection error: " << e.what() << std::endl;
      return;
    }

    zmqThreadRunning = true;
    zmqThread = std::thread(&ControlApp::zmqReceiveLoop, this);
  }

  void ControlApp::zmqReceiveLoop() {
    zmq::message_t message;
    nlohmann::json metadata; // Declare metadata here to be in scope for the whole loop

    while (zmqThreadRunning) {
      try {
        // Receive topic
        zmq::recv_result_t result = zmqSocket->recv(message, zmq::recv_flags::none);
        if (!result) {
          // Handle error or timeout
          continue;
        }
        std::string topic_str = message.to_string();

        // Receive metadata
        result = zmqSocket->recv(message, zmq::recv_flags::none);
        if (!result) {
          continue;
        }
        std::string metadata_str(message.data<char>(), message.size());
        std::cout << "[ZMQ] Received raw metadata: " << metadata_str << std::endl; // Debug print

        // Receive IPC handle
        zmq::message_t ipc_handle_msg; // Declare ipc_handle_msg here
        result = zmqSocket->recv(ipc_handle_msg, zmq::recv_flags::none);
        if (!result) {
          continue;
        }

        try {
          metadata = nlohmann::json::parse(metadata_str);

          if (metadata.count("name") == 0 || !metadata["name"].is_string() ||
              metadata.count("shape") == 0 || !metadata["shape"].is_array() ||
              metadata.count("dtype") == 0 || !metadata["dtype"].is_string() ||
              metadata.count("size") == 0 || !metadata["size"].is_number()) {
            std::cerr << "[ZMQ] Received malformed metadata for topic: " << topic_str << std::endl;
            continue;
          }

          std::lock_guard<std::mutex> lock(dataMutex);
          GpuBufferInfo &info = gpuBuffers[topic_str];
          info.shape = metadata["shape"].get<std::vector<int>>();
          info.dtype = metadata["dtype"].get<std::string>();
          info.size = metadata["size"].get<uint32_t>(); // Use uint32_t as per GpuBufferInfo
          info.ipc_handle.assign(ipc_handle_msg.data<char>(), ipc_handle_msg.data<char>() + ipc_handle_msg.size());

          // Extract CUDA UUID from metadata if available
          if (metadata.contains("cuda_uuid")) {
            info.cuda_uuid = metadata["cuda_uuid"].get<std::string>();
          } else {
            info.cuda_uuid = ""; // Clear if not present
          }

          info.updated = true;
          std::cout << "[ZMQ] Successfully received and parsed data for topic: " << topic_str << std::endl;
        }
        catch (const nlohmann::json::parse_error &e) {
          std::cerr << "[ZMQ] JSON parse error for topic " << topic_str << ": " << e.what() << std::endl;
        }
        catch (const nlohmann::json::exception &e) {
          std::cerr << "[ZMQ] JSON access error for topic " << topic_str << ": " << e.what() << std::endl;
        }
      }
      catch (const zmq::error_t &e) {
        if (zmqThreadRunning) { // Only print if not intentionally shutting down
          std::cerr << "[ZMQ] ZMQ error in receive loop: " << e.what() << std::endl;
        }
      }
      catch (const std::exception &e) {
        std::cerr << "[ZMQ] Standard exception in receive loop: " << e.what() << std::endl;
      }
      catch (...) {
        std::cerr << "[ZMQ] Unknown exception in receive loop." << std::endl;
      }
    }
  }

  void ControlApp::run() {
    auto currentTime = std::chrono::high_resolution_clock::now();

    // Get Vulkan Physical Device UUID once
    std::string vulkan_uuid_str;
    for (uint8_t byte : lveDevice.getVulkanDeviceUUID()) {
      std::stringstream ss;
      ss << std::hex << std::setw(2) << std::setfill('0') << (int)byte;
      vulkan_uuid_str += ss.str();
    }
    std::cout << "[Vulkan] Physical Device UUID: " << vulkan_uuid_str << std::endl;

    while (!lveWindow.shouldClose()) {
      glfwPollEvents();

      auto newTime = std::chrono::high_resolution_clock::now();
      float frameTime = std::chrono::duration<float, std::chrono::seconds::period>(newTime - currentTime).count();
      currentTime = newTime;

      // and create/update Vulkan resources before rendering.
      {
        std::lock_guard<std::mutex> lock(dataMutex);
        for (auto &pair : gpuBuffers) {
          if (pair.second.updated) {
            std::cout << "[Vulkan] Processing update for buffer '" << pair.first << "'..." << std::endl;

            if (pair.second.ipc_handle.empty()) {
              std::cerr << "[Vulkan] Received empty IPC handle for " << pair.first << std::endl;
              pair.second.updated = false;
              continue;
            }

            // Compare CUDA UUID with Vulkan Device UUID
            if (!pair.second.cuda_uuid.empty() && vulkan_uuid_str != pair.second.cuda_uuid) {
              std::cerr << "[Vulkan] Device UUID mismatch for buffer '" << pair.first << "'. Vulkan: " << vulkan_uuid_str << ", CUDA: " << pair.second.cuda_uuid << std::endl;
              pair.second.updated = false;
              continue;
            }

            // TODO: This logic should handle buffer recreation if size/properties change.
            // For now, we only handle initial creation.
            if (vulkanBuffers.find(pair.first) == vulkanBuffers.end()) {
              GpuBufferInfo &info = pair.second;

              VkExternalMemoryBufferCreateInfo externalMemoryBufferInfo{};
              externalMemoryBufferInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
              externalMemoryBufferInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

              VkBufferCreateInfo bufferInfo{};
              bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
              bufferInfo.pNext = &externalMemoryBufferInfo;
              bufferInfo.size = info.size; // Use the pre-calculated total size in bytes
              bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
              bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

              VulkanBuffer newVulkanBuffer;
              if (vkCreateBuffer(lveDevice.device(), &bufferInfo, nullptr, &newVulkanBuffer.buffer) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create buffer for external memory!");
              }

              // 2. Get memory requirements, including dedicated allocation info
              VkMemoryRequirements2 memRequirements2{};
              memRequirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
              VkBufferMemoryRequirementsInfo2 bufferMemReqs2{};
              bufferMemReqs2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
              bufferMemReqs2.buffer = newVulkanBuffer.buffer;
              vkGetBufferMemoryRequirements2(lveDevice.device(), &bufferMemReqs2, &memRequirements2);

              // 3. Find the required memory type index that supports external memory import
              // 4. Allocate memory for the buffer, importing from the handle
              VkImportMemoryWin32HandleInfoKHR importInfo = {};
              importInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
              importInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
              importInfo.handle = *reinterpret_cast<HANDLE *>(info.ipc_handle.data());

              // If VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT is set, we need dedicated allocation
              VkMemoryDedicatedAllocateInfo dedicatedAllocInfo{};
              dedicatedAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
              dedicatedAllocInfo.buffer = newVulkanBuffer.buffer;
              dedicatedAllocInfo.pNext = importInfo.pNext; // Chain existing pNext from importInfo
              importInfo.pNext = &dedicatedAllocInfo;      // Chain dedicatedAllocInfo after importInfo

              VkMemoryAllocateInfo allocInfo = {};
              allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
              allocInfo.pNext = &importInfo; // allocInfo now points to importInfo, which points to dedicatedAllocInfo
              allocInfo.allocationSize = memRequirements2.memoryRequirements.size;
              allocInfo.memoryTypeIndex = lveDevice.findMemoryType(
                  memRequirements2.memoryRequirements.memoryTypeBits,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, // Prefer device local memory
                  VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT);

              if (allocInfo.memoryTypeIndex == UINT32_MAX) {
                throw std::runtime_error("Failed to find suitable memory type for external import!");
              }

              std::cout << "[Vulkan] Attempting to allocate memory with memoryTypeIndex: " << allocInfo.memoryTypeIndex << ", allocationSize: " << allocInfo.allocationSize << ", handle: " << importInfo.handle << std::endl;
              std::cout << "[Vulkan] Attempting to allocate memory with memoryTypeIndex: " << allocInfo.memoryTypeIndex << ", allocationSize: " << allocInfo.allocationSize << ", handle: " << importInfo.handle << std::endl;
              VkResult allocResult = vkAllocateMemory(lveDevice.device(), &allocInfo, nullptr, &newVulkanBuffer.memory);
              if (allocResult != VK_SUCCESS) {
                std::cerr << "[Vulkan] vkAllocateMemory failed with result: " << allocResult << std::endl;
                throw std::runtime_error("Failed to allocate (import) external memory!");
              }

              // 5. Bind the buffer to the imported memory
              vkBindBufferMemory(lveDevice.device(), newVulkanBuffer.buffer, newVulkanBuffer.memory, 0);

              // 6. Allocate and update descriptor set for this buffer
              VkDescriptorSetAllocateInfo setAllocInfo{};
              setAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
              setAllocInfo.descriptorPool = globalPool->getDescriptorPool();
              setAllocInfo.descriptorSetCount = 1;
              setAllocInfo.pSetLayouts = &simple2DRenderSystem->getDescriptorSetLayout(); // Need a getter

              if (vkAllocateDescriptorSets(lveDevice.device(), &setAllocInfo, &newVulkanBuffer.descriptorSet) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate descriptor set!");
              }

              VkDescriptorBufferInfo bufferDescriptorInfo{};
              bufferDescriptorInfo.buffer = newVulkanBuffer.buffer;
              bufferDescriptorInfo.offset = 0;
              bufferDescriptorInfo.range = VK_WHOLE_SIZE;

              VkWriteDescriptorSet descriptorWrite{};
              descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
              descriptorWrite.dstSet = newVulkanBuffer.descriptorSet;
              descriptorWrite.dstBinding = 0;
              descriptorWrite.dstArrayElement = 0;
              descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
              descriptorWrite.descriptorCount = 1;
              descriptorWrite.pBufferInfo = &bufferDescriptorInfo;

              vkUpdateDescriptorSets(lveDevice.device(), 1, &descriptorWrite, 0, nullptr);

              vulkanBuffers[pair.first] = newVulkanBuffer;
              std::cout << "[Vulkan] Successfully imported and created descriptor set for buffer '" << pair.first << "'" << std::endl;
            }

            pair.second.updated = false;
          }
        }
      }

      if (auto commandBuffer = lveRenderer.beginFrame()) {
        // We don't have any objects to render yet, but the basic frame structure is kept.
        // The command buffer is ready to record rendering commands.

        lveRenderer.beginSwapChainRenderPass(commandBuffer);

        // --- 2D drawing command ---
        // Render all available buffers
        for (auto const &[name, buffer] : vulkanBuffers) {
          if (gpuBuffers.count(name)) {
            uint32_t numNeurons = gpuBuffers.at(name).shape[0];
            simple2DRenderSystem->render(commandBuffer, buffer.descriptorSet, numNeurons);
          }
        }

        lveRenderer.endSwapChainRenderPass(commandBuffer);
        lveRenderer.endFrame();
      }
    }

    vkDeviceWaitIdle(lveDevice.device());
  }

} // namespace lve