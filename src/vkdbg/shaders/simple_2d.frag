#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragLocalPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Calculate distance from the center of the quad (0,0 in local space)
    // If the distance is greater than 1.0 (the radius of our circle), discard the pixel.
    if (length(fragLocalPos) > 1.0) {
        discard;
    }

    outColor = vec4(fragColor, 1.0);
}