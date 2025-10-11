"""
End-to-End Test: Korean Text -> Embedding -> Spikes -> SNN Layer

Usage:
    python -X utf8 test/test_text_snn_pipeline.py
"""

import os
import sys
import numpy as np
import cupy as cp

# Add the project root to the Python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.text.tokenizer import KoreanTokenizer
from src.text.encoder import WordVectorEncoder
from src.text.embedding import EmbeddingLayer
from src.pyramidal_layer import PyramidalLayer, DTYPE

def get_pipeline_components(corpus: list[str], embedding_dim: int) -> tuple:
    """Initializes and returns the full text processing pipeline."""
    print("--- 1. Building Text Processing Pipeline ---")
    
    # Tokenizer
    tokenizer = KoreanTokenizer()
    print("Tokenizer initialized.")

    # Encoder (builds vocab)
    encoder = WordVectorEncoder(tokenizer=tokenizer)
    encoder.build_vocab(corpus, min_freq=1)
    print(f"Encoder initialized. Vocabulary size: {len(encoder.word_to_id)}")

    # Embedding Layer
    vocab_size = len(encoder.word_to_id)
    embedding_layer = EmbeddingLayer(vocab_size, embedding_dim)
    print(f"Embedding layer initialized for {vocab_size} words -> {embedding_dim}-dim vectors.")
    
    return tokenizer, encoder, embedding_layer

def encode_to_spikes(embedding_vector: np.ndarray, k: int = 15) -> cp.ndarray:
    """
    Converts a dense embedding vector into a binary spike vector using top-k selection.
    
    :param embedding_vector: The dense vector from the embedding layer.
    :param k: The number of neurons to spike.
    :return: A binary (0.0 or 1.0) spike vector on the GPU.
    """
    if k <= 0:
        return cp.zeros_like(embedding_vector, dtype=DTYPE)
    
    # Find the indices of the top k largest values
    # np.argpartition is faster than np.argsort for finding top-k
    top_k_indices = np.argpartition(embedding_vector, -k)[-k:]
    
    # Create a spike vector of zeros and set the top-k positions to 1.0
    spike_vector = np.zeros_like(embedding_vector, dtype=np.float32)
    spike_vector[top_k_indices] = 1.0
    
    return cp.asarray(spike_vector, dtype=DTYPE)

def main():
    """Main function to run the entire pipeline."""
    if not cp.cuda.is_available():
        raise RuntimeError("A CUDA-enabled GPU is required to run this script.")

    # --- Parameters ---
    EMBEDDING_DIM = 128  # Dimensionality of word vectors, must match SNN input
    SNN_NEURONS = 50     # Number of neurons in our SNN layer
    MAX_LEN = 15         # Max sequence length for padding

    # --- 1. Setup the Text Processing Pipeline ---
    corpus_texts = [
        "엘런 프로젝트는 뇌의 작동 방식을 모방하는 새로운 인공지능 모델입니다.",
        "스파이킹 신경망은 이벤트 기반으로 작동하여 효율적입니다.",
        "한국어 문장을 분석하여 의미를 학습하는 것이 최종 목표입니다.",
    ]
    _tokenizer, encoder, embedding_layer = get_pipeline_components(corpus_texts, EMBEDDING_DIM)

    # --- 2. Setup the SNN Layer ---
    print("\n--- 2. Initializing SNN Layer ---")
    snn_layer = PyramidalLayer(
        num_neurons=SNN_NEURONS,
        num_afferents=EMBEDDING_DIM, # Must match embedding dimension
        num_branches=5,
        random_state=2025
    )
    print(f"PyramidalLayer initialized with {snn_layer.num_afferents} inputs and {snn_layer.num_neurons} neurons.")

    # --- 3. Prepare Input Sentence ---
    print("\n--- 3. Processing Input Sentence ---")
    test_sentence = "엘런의 새로운 스파이킹 모델을 테스트합니다."
    
    # Text -> Padded IDs
    encoded_ids = encoder.encode(test_sentence, max_length=MAX_LEN)
    print(f"Original sentence: '{test_sentence}'")
    print(f"Encoded IDs (padded): {encoded_ids}")

    # Padded IDs -> Embedding Vectors
    embedding_vectors_np = embedding_layer.forward(encoded_ids)
    print(f"Embedding vectors created with shape: {embedding_vectors_np.shape}")

    # --- 4. Run the SNN Simulation ---
    print("\n--- 4. Running SNN Simulation (Time-step by Time-step) ---")
    
    snn_layer.reset_state()
    base_current = 1.5 # Provide a baseline stimulus to make neurons more excitable
    external_currents = cp.full(snn_layer.num_neurons, base_current, dtype=DTYPE)
    total_spikes = 0

    # Loop through the sequence of embedding vectors (one vector per time-step)
    for t, vector_np in enumerate(embedding_vectors_np):
        # Skip processing for pure padding vectors
        if np.all(vector_np == 0):
            print(f"Time-step {t+1:02d}: Input is padding. Skipping simulation step.")
            continue

        # Step 4a: Convert embedding vector to spike vector (using k=5)
        presynaptic_spikes = encode_to_spikes(vector_np, k=5)
        
        # Step 4b: Run one time-step of the SNN simulation
        result = snn_layer.step(presynaptic_spikes, external_currents=external_currents)
        
        # Get the spike output of the layer
        output_spikes = result.axon_spikes.get() # .get() moves data from GPU to CPU (numpy)
        
        num_spikes = np.sum(output_spikes)
        total_spikes += num_spikes

        print(f"Time-step {t+1:02d}: Input vector processed. Output spikes: {num_spikes} / {SNN_NEURONS}")
        if num_spikes > 0:
            # Find which neurons spiked
            spiking_neurons = np.where(output_spikes == 1)[0]
            print(f"  -> Spiking neurons: {spiking_neurons}")

    print("\n--- Simulation Complete ---")
    print(f"Total spikes generated during the simulation: {total_spikes}")

if __name__ == "__main__":
    main()
