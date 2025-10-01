#include "snn_data_reader.hpp"

#include <iostream>
#include <stdexcept>

namespace lve {

  SnnDataReader::SnnDataReader(const std::string &name)
      : shmName_(name) { connect(); }

  SnnDataReader::~SnnDataReader() { disconnect(); }

  void SnnDataReader::connect() {
    hMapFile_ = OpenFileMappingA(FILE_MAP_READ, FALSE, shmName_.c_str());
    if (hMapFile_ == NULL) {
      throw std::runtime_error("Failed to open shared memory file mapping.");
    }
    pBuf_ = MapViewOfFile(hMapFile_, FILE_MAP_READ, 0, 0, 0);
    if (pBuf_ == NULL) {
      CloseHandle(hMapFile_);
      throw std::runtime_error("Could not map view of file.");
    }
  }

  void SnnDataReader::disconnect() {
    if (pBuf_ != nullptr)
      UnmapViewOfFile(pBuf_);
    if (hMapFile_ != nullptr)
      CloseHandle(hMapFile_);
  }

  bool SnnDataReader::checkForNewData() {
    if (pBuf_ == nullptr) { return false; }
    uint64_t currentVersion = *static_cast<uint64_t *>(pBuf_);
    if (currentVersion != lastVersion_) {
      lastVersion_ = currentVersion;
      char *ptr = static_cast<char *>(pBuf_);
      size_t offset = 0;
      if (numLayers_ == 0) {
        offset += sizeof(uint64_t);
        numLayers_ = *reinterpret_cast<int *>(ptr + offset);
        offset += sizeof(int);
        numConnections_ = *reinterpret_cast<int *>(ptr + offset);
        offset += sizeof(int);
        numInputSynapses_ = *reinterpret_cast<int *>(ptr + offset);
        offset += sizeof(int);
        neuronsPerLayer_.resize(numLayers_);
        memcpy(neuronsPerLayer_.data(), ptr + offset, numLayers_ * sizeof(int));
        offset += numLayers_ * sizeof(int);
        int totalNeurons = getTotalNeurons();
        neuronStates_.resize(totalNeurons);
        competitionValues_.resize(totalNeurons);
        sourceIndices_.resize(numConnections_);
        targetIndices_.resize(numConnections_);
        weights_.resize(numConnections_);
        inputTargetIndices_.resize(numInputSynapses_);
        inputWeights_.resize(numInputSynapses_);
        memcpy(neuronStates_.data(), ptr + offset, totalNeurons * sizeof(float));
        offset += totalNeurons * sizeof(float);
        memcpy(competitionValues_.data(), ptr + offset, totalNeurons * sizeof(float));
        offset += totalNeurons * sizeof(float);
        memcpy(sourceIndices_.data(), ptr + offset, numConnections_ * sizeof(int));
        offset += numConnections_ * sizeof(int);
        memcpy(targetIndices_.data(), ptr + offset, numConnections_ * sizeof(int));
        offset += numConnections_ * sizeof(int);
        memcpy(weights_.data(), ptr + offset, numConnections_ * sizeof(float));
        offset += numConnections_ * sizeof(float);
        memcpy(inputTargetIndices_.data(), ptr + offset, numInputSynapses_ * sizeof(int));
        offset += numInputSynapses_ * sizeof(int);
        memcpy(inputWeights_.data(), ptr + offset, numInputSynapses_ * sizeof(float));
      } else {
        int totalNeurons = getTotalNeurons();
        size_t base_offset = sizeof(uint64_t) + sizeof(int) * 3 + numLayers_ * sizeof(int);
        memcpy(neuronStates_.data(), ptr + base_offset, totalNeurons * sizeof(float));
        memcpy(competitionValues_.data(), ptr + base_offset + totalNeurons * sizeof(float), totalNeurons * sizeof(float));
        size_t weights_offset = base_offset + totalNeurons * sizeof(float) * 2 + numConnections_ * sizeof(int) * 2;
        memcpy(weights_.data(), ptr + weights_offset, numConnections_ * sizeof(float));
        size_t input_weights_offset = weights_offset + numConnections_ * sizeof(float) + numInputSynapses_ * sizeof(int);
        memcpy(inputWeights_.data(), ptr + input_weights_offset, numInputSynapses_ * sizeof(float));
      }
      return true;
    }
    return false;
  }

} // namespace lve