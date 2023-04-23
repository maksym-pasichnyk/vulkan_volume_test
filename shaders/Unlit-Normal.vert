#version 450

layout(push_constant) uniform ShaderUniforms {
    mat4 world_to_clip;
    vec3 camera_position;
};

layout(location = 0) in vec3 in_vert_position;
layout(location = 1) in vec3 in_vert_normal;
layout(location = 2) in vec2 in_vert_texcoord;

layout(location = 0) out vec3 out_vert_position;
layout(location = 1) out vec2 out_vert_texcoord;
layout(location = 2) out vec3 out_vert_direction;

struct Vertex {
    vec3 position;
    vec3 color;
    vec2 texcoord;
};

void main() {
    vec2 vert_texcoord = in_vert_texcoord;
    vec3 vert_position = in_vert_position * vec3(48, 12, 112);

    gl_Position = world_to_clip * vec4(vert_position, 1.0);
    out_vert_position = vert_position;
    out_vert_texcoord = vert_texcoord;
    out_vert_direction = (vert_position - camera_position);
}