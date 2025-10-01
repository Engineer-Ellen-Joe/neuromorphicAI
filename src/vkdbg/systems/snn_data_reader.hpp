#pragma once

#include <windows.h>
#include <string>
#include <vector>
#include <cstdint>
#include <numeric> // For std::accumulate

namespace lve {

  class SnnDataReader {
  public:
    SnnDataReader(const std::string &shmName);
    ~SnnDataReader();

    SnnDataReader(const SnnDataReader &) = delete;
    SnnDataReader &operator=(const SnnDataReader &) = delete;

    bool checkForNewData();

    const std::vector<float> &getNeuronStates() const { return neuronStates_; }
    int getNumLayers() const { return numLayers_; }
    const std::vector<int> &getNeuronsPerLayer() const { return neuronsPerLayer_; }
    int getTotalNeurons() const {
        return std::accumulate(neuronsPerLayer_.begin(), neuronsPerLayer_.end(), 0);
    }

    // Synapse data accessors
    int getNumConnections() const { return numConnections_; }
    const std::vector<int>& getSourceIndices() const { return sourceIndices_; }
    const std::vector<int>& getTargetIndices() const { return targetIndices_; }
    const std::vector<float>& getWeights() const { return weights_; }

    // Input Synapse data accessors
    int getNumInputSynapses() const { return numInputSynapses_; }
    const std::vector<int>& getInputTargetIndices() const { return inputTargetIndices_; }
    const std::vector<float>& getInputWeights() const { return inputWeights_; }

    // Additional neuron data
    const std::vector<float>& getCompetitionValues() const { return competitionValues_; }

  private:
    void connect();
    void disconnect();

    std::string shmName_;
    size_t bufferSize_ = 0; 

    HANDLE hMapFile_ = nullptr;
    void *pBuf_ = nullptr;

    std::vector<float> neuronStates_;
    uint64_t lastVersion_ = -1;

    // SNN structure info
    int numLayers_ = 0;
    std::vector<int> neuronsPerLayer_;

    // Synapse info
    int numConnections_ = 0;
    std::vector<int> sourceIndices_;
    std::vector<int> targetIndices_;
    std::vector<float> weights_;

    // Input Synapse info
    int numInputSynapses_ = 0;
    std::vector<int> inputTargetIndices_;
    std::vector<float> inputWeights_;

    // Additional neuron data
    std::vector<float> competitionValues_;
  };

} // namespace lve