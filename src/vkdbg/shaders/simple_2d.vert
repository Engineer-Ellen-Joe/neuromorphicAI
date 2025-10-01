#version 450

layout(location = 0) in vec2 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec2 local_pos;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragLocalPos;

layout(push_constant) uniform Push {
    mat4 proj;
    mat4 view;
} push;

void main() {
    gl_Position = push.proj * push.view * vec4(position, 0.0, 1.0);
    fragColor = color;
    fragLocalPos = local_pos;
}
