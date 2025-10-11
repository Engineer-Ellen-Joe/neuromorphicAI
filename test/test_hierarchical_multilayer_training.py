"""

Usage:
    python -X utf8 test/hierarchical_multilayer_training.py
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import cupy as cp

# Ensure the repository root (containing src/) is on the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.pyramidal_layer import PyramidalLayer, DTYPE


@dataclass
class LayerDiagnostics:
    macro_index: int
    sublayer_index: int
    neuron_count: int
    input_max_delta: float
    input_mean_delta: float
    output_max_delta: float
    output_mean_delta: float
    average_firing_rate: float
    total_spikes: int


def _partition_neurons(total: int) -> List[int]:
    """
    Partition a macro layer neuron count into three sublayers.

    The ratios are biased toward earlier sublayers to ensure that the macro layer
    has a broad receptive field while still allowing deep specialization.
    """
    ratios = np.array([0.45, 0.33, 0.22], dtype=np.float64)
    counts = np.floor(ratios * total).astype(int)
    remainder = int(total - counts.sum())

    # Distribute remaining neurons to the largest deficits first.
    order = np.argsort(-ratios)
    for idx in order:
        if remainder <= 0:
            break
        counts[idx] += 1
        remainder -= 1

    # Correct any rounding-induced overflow.
    idx = len(counts) - 1
    while counts.sum() > total:
        if counts[idx] > 1:
            counts[idx] -= 1
        idx = (idx - 1) % len(counts)

    return counts.tolist()


def _build_macro_layer(total_neurons: int, incoming_dim: int, seed: int) -> List[PyramidalLayer]:
    """
    Construct a macro layer as a sequence of sublayers. Each subsequent sublayer receives
    concatenated activations from all previous sublayers, enabling rich intra-layer connectivity.
    """
    sublayer_sizes = _partition_neurons(total_neurons)
    sublayers: List[PyramidalLayer] = []
    accumulated_outputs = 0

    for idx, size in enumerate(sublayer_sizes):
        if idx == 0:
            afferents = incoming_dim
        else:
            afferents = incoming_dim + accumulated_outputs

        layer = PyramidalLayer(
            num_neurons=size,
            num_afferents=afferents,
            num_branches=3,
            dt=1.0,
            input_learning_rate=0.04,
            output_learning_rate=0.02,
            stdp_a_plus=0.035,
            stdp_a_minus=0.015,
            stdp_tau_pre=18.0,
            stdp_tau_post=22.0,
            branch_activity_tau=70.0,
            bcm_tau=950.0,
            axon_spike_current=7.5,
            conduction_threshold=0.0,
            dendritic_time_constant=32.0,
            dendritic_capacitance=0.11,
            input_weight_bounds=(0.0, 6.5),
            output_weight_bounds=(0.0, 6.5),
            random_state=seed * 97 + idx * 13,
        )

        sublayers.append(layer)
        accumulated_outputs += size

    return sublayers


def _generate_stimuli(input_dim: int, *, rng_seed: int = 11) -> List[cp.ndarray]:
    """
    Create a diverse corpus of stimuli consisting of:
    - unit vectors,
    - overlapping multi-hot patterns,
    - sinusoidal wavefronts,
    - random sparse and dense bursts.
    """
    rng = np.random.default_rng(rng_seed)
    patterns: List[np.ndarray] = []

    # One-hot baseline patterns.
    for idx in range(min(input_dim, 24)):
        vec = np.zeros(input_dim, dtype=np.float32)
        vec[idx] = 1.0
        patterns.append(vec)

    # Multi-hot clusters.
    for size in (3, 5, 7):
        for _ in range(10):
            idxs = rng.choice(input_dim, size=size, replace=False)
            vec = np.zeros(input_dim, dtype=np.float32)
            vec[idxs] = 1.0
            patterns.append(vec)

    # Sinusoidal and triangular waveforms, thresholded to spike-like patterns.
    domain = np.linspace(0.0, 2.0 * np.pi, input_dim, dtype=np.float32)
    for freq in (1.0, 1.7, 2.5, 3.1):
        waveform = 0.5 * (np.sin(domain * freq) + 1.0)
        patterns.append((waveform > 0.65).astype(np.float32))
        triangle = np.abs((domain * freq / np.pi) % 2 - 1)
        patterns.append((triangle > 0.55).astype(np.float32))

    # Random bursts with varying sparsity.
    for density in (0.15, 0.25, 0.4):
        bursts = rng.random((12, input_dim), dtype=np.float32)
        patterns.extend((bursts < density).astype(np.float32))

    # Deduplicate exact duplicates while preserving order.
    seen = set()
    unique_patterns: List[np.ndarray] = []
    for vec in patterns:
        key = tuple(vec.tolist())
        if key not in seen:
            seen.add(key)
            unique_patterns.append(vec)

    return [cp.asarray(vec, dtype=DTYPE) for vec in unique_patterns]


def _reset_learning_rates(layers: Iterable[PyramidalLayer], *, enable: bool) -> Dict[int, Tuple[float, float]]:
    """
    Enable or disable learning by adjusting input/output learning rates.
    Returns a dictionary mapping layer id to the original learning rates.
    """
    stored: Dict[int, Tuple[float, float]] = {}

    for layer in layers:
        key = id(layer)
        if enable:
            if key in stored:
                layer.input_learning_rate, layer.output_learning_rate = stored[key]
        else:
            stored[key] = (layer.input_learning_rate, layer.output_learning_rate)
            layer.input_learning_rate = 0.0
            layer.output_learning_rate = 0.0

    return stored


def _iterate_layers(macro_layers: Sequence[Sequence[PyramidalLayer]]) -> Iterable[PyramidalLayer]:
    for macro in macro_layers:
        for layer in macro:
            yield layer


def train_hierarchical_network(
    macro_layers: List[List[PyramidalLayer]],
    stimuli: Sequence[cp.ndarray],
    *,
    epochs: int,
    base_current: float = 7.5,
) -> None:
    """
    Execute multi-epoch training across the macro hierarchy.

    Each sublayer receives a concatenation of the macro input and outputs of all preceding
    sublayers within the same macro layer, ensuring dense intra-layer feedback.
    """
    external_currents: List[List[cp.ndarray]] = [
        [cp.full(layer.num_neurons, base_current, dtype=DTYPE) for layer in macro]
        for macro in macro_layers
    ]

    for layer in _iterate_layers(macro_layers):
        layer.reset_state()

    for epoch in range(epochs):
        for stimulus in stimuli:
            macro_input = stimulus
            for macro, currents in zip(macro_layers, external_currents):
                sublayer_outputs: List[cp.ndarray] = []
                base_signal = macro_input

                for layer, current in zip(macro, currents):
                    if sublayer_outputs:
                        concatenated = cp.concatenate([base_signal] + sublayer_outputs)
                    else:
                        concatenated = base_signal

                    concatenated = cp.clip(concatenated, 0.0, 1.0)
                    result = layer.step(concatenated, external_currents=current)
                    sublayer_outputs.append(result.axon_spikes)

                macro_input = cp.concatenate(sublayer_outputs)

        # Periodically decay internal states to avoid runaway potentials.
        if (epoch + 1) % 2 == 0:
            for layer in _iterate_layers(macro_layers):
                layer.reset_state()


def evaluate_network_responses(
    macro_layers: List[List[PyramidalLayer]],
    stimuli: Sequence[cp.ndarray],
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Run a frozen forward pass to collect activity fingerprints for diagnostics.
    """
    all_layers = list(_iterate_layers(macro_layers))
    saved_rates = _reset_learning_rates(all_layers, enable=False)

    try:
        for layer in all_layers:
            layer.reset_state()

        responses: List[np.ndarray] = []
        spike_totals: List[int] = []

        for stimulus in stimuli:
            macro_input = stimulus
            intermediate_outputs: List[cp.ndarray] = []

            for macro in macro_layers:
                sublayer_outputs: List[cp.ndarray] = []
                for layer in macro:
                    if sublayer_outputs:
                        afferents = cp.concatenate([macro_input] + sublayer_outputs)
                    else:
                        afferents = macro_input
                    afferents = cp.clip(afferents, 0.0, 1.0)
                    result = layer.step(afferents, external_currents=cp.zeros(layer.num_neurons, dtype=DTYPE))
                    sublayer_outputs.append(result.axon_spikes)

                macro_input = cp.concatenate(sublayer_outputs)
                intermediate_outputs.append(macro_input)

            # Convert GPU arrays to CPU for easier analysis.
            concatenated = cp.concatenate(intermediate_outputs).get()
            responses.append(concatenated)
            spike_totals.append(int(np.sum(concatenated)))

        return responses, spike_totals
    finally:
        for layer in all_layers:
            lr_in, lr_out = saved_rates[id(layer)]
            layer.input_learning_rate = lr_in
            layer.output_learning_rate = lr_out


def collect_diagnostics(
    macro_layers: List[List[PyramidalLayer]],
    input_snapshots: Sequence[Sequence[cp.ndarray]],
    output_snapshots: Sequence[Sequence[cp.ndarray]],
    spike_log: Dict[Tuple[int, int], List[int]],
) -> List[LayerDiagnostics]:
    diagnostics: List[LayerDiagnostics] = []

    for macro_idx, (macro, snap_in_macro, snap_out_macro) in enumerate(
        zip(macro_layers, input_snapshots, output_snapshots)
    ):
        for sub_idx, (layer, snap_in, snap_out) in enumerate(
            zip(macro, snap_in_macro, snap_out_macro)
        ):
            delta_in = cp.abs(layer.input_weights - snap_in)
            delta_out = cp.abs(layer.output_weights - snap_out)
            spikes = spike_log.get((macro_idx, sub_idx), [0])

            diagnostics.append(
                LayerDiagnostics(
                    macro_index=macro_idx,
                    sublayer_index=sub_idx,
                    neuron_count=layer.num_neurons,
                    input_max_delta=float(cp.max(delta_in).get()),
                    input_mean_delta=float(cp.mean(delta_in).get()),
                    output_max_delta=float(cp.max(delta_out).get()),
                    output_mean_delta=float(cp.mean(delta_out).get()),
                    average_firing_rate=float(np.mean(spikes)),
                    total_spikes=int(np.sum(spikes)),
                )
            )

    return diagnostics


def _log_training_activity(
    macro_layers: List[List[PyramidalLayer]],
    stimuli: Sequence[cp.ndarray],
    *,
    monitor_samples: int = 20,
) -> Dict[Tuple[int, int], List[int]]:
    """
    Run a brief monitored pass to gather spike counts per sublayer for diagnostics.
    """
    spike_log: Dict[Tuple[int, int], List[int]] = {}
    for layer in _iterate_layers(macro_layers):
        layer.reset_state()

    samples = stimuli[:monitor_samples]

    for stimulus in samples:
        macro_input = stimulus
        for macro_idx, macro in enumerate(macro_layers):
            sublayer_outputs: List[cp.ndarray] = []
            for sub_idx, layer in enumerate(macro):
                if sublayer_outputs:
                    afferents = cp.concatenate([macro_input] + sublayer_outputs)
                else:
                    afferents = macro_input
                afferents = cp.clip(afferents, 0.0, 1.0)
                result = layer.step(afferents, external_currents=cp.zeros(layer.num_neurons, dtype=DTYPE))
                sublayer_outputs.append(result.axon_spikes)

                key = (macro_idx, sub_idx)
                spike_log.setdefault(key, []).append(int(cp.sum(result.axon_spikes).get()))

            macro_input = cp.concatenate(sublayer_outputs)

    return spike_log


def assert_learning_occurred(diagnostics: Sequence[LayerDiagnostics]) -> None:
    """
    Verify that every sublayer exhibited meaningful synaptic plasticity.
    """
    for diag in diagnostics:
        if diag.input_max_delta < 5e-4:
            raise RuntimeError(
                f"Insufficient input-weight plasticity in macro {diag.macro_index + 1}, "
                f"sublayer {diag.sublayer_index + 1}: max Δw={diag.input_max_delta:.6f}"
            )
        if diag.output_max_delta < 5e-4:
            raise RuntimeError(
                f"Insufficient output-weight plasticity in macro {diag.macro_index + 1}, "
                f"sublayer {diag.sublayer_index + 1}: max Δw={diag.output_max_delta:.6f}"
            )


def summarize_responses(responses: Sequence[np.ndarray]) -> None:
    """
    Print high-level statistics of network responses to confirm pattern diversity.
    """
    matrix = np.vstack(responses)
    variance_per_pattern = np.var(matrix, axis=1)
    if np.allclose(variance_per_pattern, 0.0):
        raise RuntimeError("All response patterns collapsed to a constant vector.")

    overall_variance = float(np.mean(variance_per_pattern))
    l2_norms = np.linalg.norm(matrix, axis=1)
    cosine_matrix = np.corrcoef(matrix)

    print("\n=== Response Statistics ===")
    print(f"Patterns assessed: {matrix.shape[0]}")
    print(f"Response dimensionality: {matrix.shape[1]}")
    print(f"Mean activation norm: {float(np.mean(l2_norms)):.4f}")
    print(f"Mean per-pattern variance: {overall_variance:.6f}")
    print(f"Cosine similarity (summary): min={np.min(cosine_matrix):.4f}, "
          f"mean={np.mean(cosine_matrix):.4f}, max={np.max(cosine_matrix):.4f}")


def main() -> None:
    if not cp.cuda.is_available():
        raise RuntimeError("CUDA-enabled GPU is required to execute this script.")

    np.random.seed(2026)
    cp.random.seed(2026)

    macro_configuration = [200, 800, 300, 100]
    input_dimension = 192

    macro_layers: List[List[PyramidalLayer]] = []
    incoming_dim = input_dimension
    for idx, total_neurons in enumerate(macro_configuration, start=1):
        macro = _build_macro_layer(total_neurons, incoming_dim, seed=idx)
        macro_layers.append(macro)
        incoming_dim = sum(layer.num_neurons for layer in macro)

    input_snapshots = [
        [layer.input_weights.copy() for layer in macro] for macro in macro_layers
    ]
    output_snapshots = [
        [layer.output_weights.copy() for layer in macro] for macro in macro_layers
    ]

    stimuli = _generate_stimuli(input_dimension)
    print(f"Generated {len(stimuli)} unique stimuli for training.")

    train_hierarchical_network(macro_layers, stimuli, epochs=7, base_current=7.8)

    spike_log = _log_training_activity(macro_layers, stimuli)
    diagnostics = collect_diagnostics(macro_layers, input_snapshots, output_snapshots, spike_log)

    print("\n=== Layer Diagnostics ===")
    for diag in diagnostics:
        print(
            f"Macro {diag.macro_index + 1} / Sublayer {diag.sublayer_index + 1} "
            f"(neurons={diag.neuron_count}): "
            f"Δw_in_max={diag.input_max_delta:.6f}, Δw_in_mean={diag.input_mean_delta:.6f}, "
            f"Δw_out_max={diag.output_max_delta:.6f}, Δw_out_mean={diag.output_mean_delta:.6f}, "
            f"avg_spikes={diag.average_firing_rate:.2f}, total_spikes={diag.total_spikes}"
        )

    assert_learning_occurred(diagnostics)

    eval_stimuli = stimuli[: max(32, len(stimuli) // 3)]
    responses, spike_totals = evaluate_network_responses(macro_layers, eval_stimuli)

    print("\nCollected response signatures for evaluation stimuli.")
    summarize_responses(responses)
    print(f"Aggregate spike totals across evaluation stimuli: min={min(spike_totals)}, "
          f"max={max(spike_totals)}, mean={sum(spike_totals) / len(spike_totals):.2f}")

    print("\nAll diagnostics passed successfully.")


if __name__ == "__main__":
    main()
