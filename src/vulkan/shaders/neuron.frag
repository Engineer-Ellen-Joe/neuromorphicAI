#version 450

// Input from vertex shader
layout(location = 0) in float inPotential;

// Output color
layout(location = 0) out vec4 outColor;

// Constants for potential range (mV)
const float REST_POTENTIAL = -70.0;
const float THRESHOLD_POTENTIAL = -50.0;

void main() {
    // Normalize the potential to a 0-1 range for color mapping
    // clamp to avoid extreme values
    float normalizedPotential = (inPotential - REST_POTENTIAL) / (THRESHOLD_POTENTIAL - REST_POTENTIAL);
    normalizedPotential = clamp(normalizedPotential, 0.0, 1.0);

    // Interpolate color from blue (rest) to red (threshold)
    vec3 color = mix(vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0), normalizedPotential);

    outColor = vec4(color, 1.0);
}
