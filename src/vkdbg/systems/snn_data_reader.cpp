#include "snn_data_reader.hpp"

#include <iostream>
#include <stdexcept>

namespace lve {

SnnDataReader::SnnDataReader(const std::string& name, size_t count)
    : shmName(name), numNeurons(count) {
    bufferSize = sizeof(uint64_t) + numNeurons * sizeof(float);
    neuronStates.resize(numNeurons);
    connect();
}

SnnDataReader::~SnnDataReader() {
    disconnect();
}

void SnnDataReader::connect() {
    std::cout << "Attempting to connect to shared memory: " << shmName << std::endl;

    hMapFile = OpenFileMappingA(
        FILE_MAP_READ,   // read access
        FALSE,           // do not inherit the name
        shmName.c_str()); // name of mapping object

    if (hMapFile == NULL) {
        throw std::runtime_error("Failed to open shared memory file mapping. Is the Python script running?");
    }

    pBuf = MapViewOfFile(
        hMapFile,       // handle to map object
        FILE_MAP_READ,  // read access
        0,
        0,
        bufferSize);

    if (pBuf == NULL) {
        CloseHandle(hMapFile);
        throw std::runtime_error("Could not map view of file.");
    }

    std::cout << "Successfully connected to shared memory." << std::endl;
}

void SnnDataReader::disconnect() {
    if (pBuf != nullptr) {
        UnmapViewOfFile(pBuf);
        pBuf = nullptr;
    }

    if (hMapFile != nullptr) {
        CloseHandle(hMapFile);
        hMapFile = nullptr;
    }
    std::cout << "Disconnected from shared memory." << std::endl;
}

bool SnnDataReader::checkForNewData() {
    if (pBuf == nullptr) {
        return false;
    }

    // Read the version number from the beginning of the buffer
    uint64_t currentVersion = *static_cast<uint64_t*>(pBuf);

    if (currentVersion != lastVersion) {
        lastVersion = currentVersion;

        // Calculate the starting address of the float data
        void* data_ptr = static_cast<char*>(pBuf) + sizeof(uint64_t);

        // Copy the neuron data into our local vector
        memcpy(neuronStates.data(), data_ptr, numNeurons * sizeof(float));

        return true;
    }

    return false;
}

} // namespace lve
