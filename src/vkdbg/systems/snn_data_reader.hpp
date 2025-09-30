#pragma once

#include <windows.h>
#include <string>
#include <vector>
#include <cstdint>

namespace lve {

class SnnDataReader {
public:
    SnnDataReader(const std::string& shmName, size_t numNeurons);
    ~SnnDataReader();

    SnnDataReader(const SnnDataReader&) = delete;
    SnnDataReader& operator=(const SnnDataReader&) = delete;

    // Checks for new data in shared memory. If found, reads it into the internal buffer.
    // Returns true if new data was read, false otherwise.
    bool checkForNewData();

    // Returns a constant reference to the most recently read neuron data.
    const std::vector<float>& getNeuronStates() const { return neuronStates; }

private:
    void connect();
    void disconnect();

    // Shared memory properties
    std::string shmName;
    size_t numNeurons;
    size_t bufferSize;

    // Win32 handles
    HANDLE hMapFile = nullptr;
    void* pBuf = nullptr;

    // Data
    std::vector<float> neuronStates;
    uint64_t lastVersion = -1;
};

} // namespace lve
