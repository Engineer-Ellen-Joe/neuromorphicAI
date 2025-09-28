#version 450

// SSBO (Storage Buffer) containing neuron data
layout(set = 0, binding = 0) readonly buffer NeuronData {
    float potentials[];
} ssbo;

// Output to fragment shader
layout(location = 0) out float outPotential;

// We will draw a grid of neurons. Let's say 20 neurons per row.
const int NEURONS_PER_ROW = 20;
const float NEURON_SPACING = 0.08; // Spacing between neurons
const float NEURON_SIZE = 0.05;      // Size of the neuron quad

void main() {
    // gl_VertexIndex will be 0, 1, 2, 3 for the first neuron, 4, 5, 6, 7 for the second, etc.
    int neuronId = gl_VertexIndex / 4; // Each neuron is a quad (4 vertices)
    int vertexId = gl_VertexIndex % 4; // 0, 1, 2, 3 for the corners of the quad

    // Calculate grid position
    float gridX = float(neuronId % NEURONS_PER_ROW);
    float gridY = float(neuronId / NEURONS_PER_ROW);

    // Center position of the neuron on the grid
    vec2 centerPos = vec2(-0.9 + gridX * NEURON_SPACING, -0.9 + gridY * NEURON_SPACING);

    // Determine corner offset based on vertexId
    vec2 offset;
    if (vertexId == 0) offset = vec2(-NEURON_SIZE, -NEURON_SIZE);
    else if (vertexId == 1) offset = vec2( NEURON_SIZE, -NEURON_SIZE);
    else if (vertexId == 2) offset = vec2(-NEURON_SIZE,  NEURON_SIZE);
    else offset = vec2( NEURON_SIZE,  NEURON_SIZE);

    // Final position of the vertex
    gl_Position = vec4(centerPos + offset, 0.0, 1.0);

    // Pass the potential of the current neuron to the fragment shader
    outPotential = ssbo.potentials[neuronId];
}
