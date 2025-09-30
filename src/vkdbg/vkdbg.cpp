// vkdbg.cpp
// Build with pybind11, link Vulkan, and link the same LveDevice (or adapt to your device wrapper).
// This is a scaffold — 네 프로젝트의 LveDevice 래퍼를 사용하도록 적절히 어댑트해야 함.

#define VK_USE_PLATFORM_WIN32_KHR
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vulkan/vulkan.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <memory>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace py = pybind11;

struct ExportedBufferInfo {
    uint64_t handle; // HANDLE as integer
    size_t allocation_size;
    size_t alignment;
    std::string handle_type; // "KMT" or "OPAQUE"
    std::string uuid;
    uint64_t internal_id;
};

class VkManager {
public:
    VkManager() {
    }

    ~VkManager() {
        std::lock_guard<std::mutex> lg(mutex_);
        for (auto &kv : buffers_) {
            auto &b = kv.second;
            vkDestroyBuffer(device_, b.buffer, nullptr);
            vkFreeMemory(device_, b.memory, nullptr);
        }
    }

    ExportedBufferInfo create_exportable_buffer(size_t requestedSize) {
        std::lock_guard<std::mutex> lg(mutex_);

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = requestedSize;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkExternalMemoryBufferCreateInfo externalBuf{};
        externalBuf.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
        externalBuf.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        bufferInfo.pNext = &externalBuf;

        VkBuffer buffer;
        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateBuffer failed");
        }

        VkMemoryRequirements memReq{};
        vkGetBufferMemoryRequirements(device_, buffer, &memReq);

        VkDeviceSize allocSize = memReq.size;
        if (memReq.alignment > 0 && (allocSize % memReq.alignment) != 0) {
            allocSize = ((allocSize + memReq.alignment - 1) / memReq.alignment) * memReq.alignment;
        }

        VkMemoryDedicatedAllocateInfo dedicatedAlloc{};
        dedicatedAlloc.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        dedicatedAlloc.buffer = buffer;
        dedicatedAlloc.image = VK_NULL_HANDLE;

        VkExportMemoryAllocateInfo exportAlloc{};
        exportAlloc.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportAlloc.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        exportAlloc.pNext = &dedicatedAlloc;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = allocSize;
        allocInfo.memoryTypeIndex = findMemoryType(memReq.memoryTypeBits,
                                                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                   VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT);
        allocInfo.pNext = &exportAlloc;

        VkDeviceMemory mem;
        if (vkAllocateMemory(device_, &allocInfo, nullptr, &mem) != VK_SUCCESS) {
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("vkAllocateMemory failed");
        }

        if (vkBindBufferMemory(device_, buffer, mem, 0) != VK_SUCCESS) {
            vkFreeMemory(device_, mem, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("vkBindBufferMemory failed");
        }

        VkMemoryGetWin32HandleInfoKHR handleInfo{};
        handleInfo.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
        handleInfo.memory = mem;
        handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        HANDLE winHandle = NULL;
        auto fpGetHandle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR");
        if (!fpGetHandle) {
            vkFreeMemory(device_, mem, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("vkGetMemoryWin32HandleKHR not found");
        }
        if (fpGetHandle(device_, &handleInfo, &winHandle) != VK_SUCCESS) {
            vkFreeMemory(device_, mem, nullptr);
            vkDestroyBuffer(device_, buffer, nullptr);
            throw std::runtime_error("vkGetMemoryWin32HandleKHR failed");
        }

        VkPhysicalDeviceIDProperties devID{};
        devID.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 devProps2{};
        devProps2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        devProps2.pNext = &devID;
        vkGetPhysicalDeviceProperties2(physicalDevice_, &devProps2);

        std::stringstream ss;
        for (int i = 0; i < VK_UUID_SIZE; ++i) {
            ss << std::hex << std::setw(2) << std::setfill('0') << (int)devID.deviceUUID[i];
        }

        uint64_t id = next_id_++;
        buffers_[id] = BufferEntry{ buffer, mem, reinterpret_cast<uint64_t>(winHandle), allocSize, (size_t)memReq.alignment };

        ExportedBufferInfo out;
        out.handle = reinterpret_cast<uint64_t>(winHandle);
        out.allocation_size = (size_t)allocSize;
        out.alignment = (size_t)memReq.alignment;
        out.handle_type = "KMT";
        out.uuid = ss.str();
        out.internal_id = id;
        return out;
    }

    void destroy_exportable_buffer(uint64_t id) {
        std::lock_guard<std::mutex> lg(mutex_);
        auto it = buffers_.find(id);
        if (it == buffers_.end()) return;
        if (it->second.handle) {
            CloseHandle((HANDLE)it->second.handle);
        }
        vkDestroyBuffer(device_, it->second.buffer, nullptr);
        vkFreeMemory(device_, it->second.memory, nullptr);
        buffers_.erase(it);
    }

    void setDevices(VkDevice dev, VkPhysicalDevice phys) {
        device_ = dev;
        physicalDevice_ = phys;
    }

private:
    struct BufferEntry {
        VkBuffer buffer;
        VkDeviceMemory memory;
        uint64_t handle;
        size_t allocation_size;
        size_t alignment;
    };

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagBits handleType) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeFilter & (1u << i)) && ( (memProps.memoryTypes[i].propertyFlags & properties) == properties)) {
                return i;
            }
        }
        throw std::runtime_error("No suitable memory type found");
    }

    std::mutex mutex_;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    std::unordered_map<uint64_t, BufferEntry> buffers_;
    uint64_t next_id_ = 1;
};

static std::unique_ptr<VkManager> g_mgr;

PYBIND11_MODULE(vkdbg, m) {
    py::class_<ExportedBufferInfo>(m, "ExportedBufferInfo")
        .def_readonly("handle", &ExportedBufferInfo::handle)
        .def_readonly("allocation_size", &ExportedBufferInfo::allocation_size)
        .def_readonly("alignment", &ExportedBufferInfo::alignment)
        .def_readonly("handle_type", &ExportedBufferInfo::handle_type)
        .def_readonly("uuid", &ExportedBufferInfo::uuid)
        .def_readonly("internal_id", &ExportedBufferInfo::internal_id);

    m.def("init_manager", []() {
        if (!g_mgr) g_mgr = std::make_unique<VkManager>();
    });

    m.def("set_devices", [](uint64_t dev, uint64_t phys){
        if (!g_mgr) g_mgr = std::make_unique<VkManager>();
        g_mgr->setDevices(reinterpret_cast<VkDevice>(dev), reinterpret_cast<VkPhysicalDevice>(phys));
    });

    m.def("create_exportable_buffer", [](size_t size)->ExportedBufferInfo {
        if (!g_mgr) g_mgr = std::make_unique<VkManager>();
        return g_mgr->create_exportable_buffer(size);
    });

    m.def("destroy_exportable_buffer", [](uint64_t id){
        if (g_mgr) g_mgr->destroy_exportable_buffer(id);
    });
}
