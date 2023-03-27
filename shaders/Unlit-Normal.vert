#version 450

layout(push_constant) uniform ShaderUniforms {
    mat4 world_to_clip;
    vec3 camera_position;
};

layout(location = 0) in vec3 in_vert_position;
layout(location = 1) in vec3 in_vert_normal;

layout(location = 0) out vec3 out_frag_color;
layout(location = 1) out vec3 out_vert_position;
layout(location = 2) out vec2 out_vert_texcoord;
layout(location = 3) out vec3 out_vert_direction;

struct Vertex {
    vec3 position;
    vec3 color;
    vec2 texcoord;
};

Vertex triangle_vertices[] = {
    // left face (white)
    {{0.0F, 0.0F, 1.0F}, {.9f, .9f, .9f}, {0.0F, 0.0F}},
    {{0.0F, 1.0F, 1.0F}, {.9f, .9f, .9f}, {0.0F, 1.0F}},
    {{0.0F, 1.0F, 0.0F}, {.9f, .9f, .9f}, {1.0F, 1.0F}},
    {{0.0F, 0.0F, 0.0F}, {.9f, .9f, .9f}, {1.0F, 0.0F}},

    // right face (yellow)
    {{1.0F, 0.0F, 0.0F}, {.8f, .8f, .1f}, {0.0F, 0.0F}},
    {{1.0F, 1.0F, 0.0F}, {.8f, .8f, .1f}, {0.0F, 1.0F}},
    {{1.0F, 1.0F, 1.0F}, {.8f, .8f, .1f}, {1.0F, 1.0F}},
    {{1.0F, 0.0F, 1.0F}, {.8f, .8f, .1f}, {1.0F, 0.0F}},

    // bottom face (orange)
    {{0.0F, 0.0F, 1.0F}, {.9f, .6f, .1f}, {0.0F, 0.0F}},
    {{0.0F, 0.0F, 0.0F}, {.9f, .6f, .1f}, {0.0F, 1.0F}},
    {{1.0F, 0.0F, 0.0F}, {.9f, .6f, .1f}, {1.0F, 1.0F}},
    {{1.0F, 0.0F, 1.0F}, {.9f, .6f, .1f}, {1.0F, 0.0F}},

    // top face (red)
    {{0.0F, 1.0F, 0.0F}, {.8f, .1f, .1f}, {0.0F, 0.0F}},
    {{0.0F, 1.0F, 1.0F}, {.8f, .1f, .1f}, {0.0F, 1.0F}},
    {{1.0F, 1.0F, 1.0F}, {.8f, .1f, .1f}, {1.0F, 1.0F}},
    {{1.0F, 1.0F, 0.0F}, {.8f, .1f, .1f}, {1.0F, 0.0F}},

    // nose face (blue)
    {{1.0F, 0.0F, 1.0F}, {.1f, .1f, .8f}, {0.0F, 0.0F}},
    {{1.0F, 1.0F, 1.0F}, {.1f, .1f, .8f}, {0.0F, 1.0F}},
    {{0.0F, 1.0F, 1.0F}, {.1f, .1f, .8f}, {1.0F, 1.0F}},
    {{0.0F, 0.0F, 1.0F}, {.1f, .1f, .8f}, {1.0F, 0.0F}},

    // tail face (green)
    {{0.0F, 0.0F, 0.0F}, {.1f, .8f, .1f}, {0.0F, 0.0F}},
    {{0.0F, 1.0F, 0.0F}, {.1f, .8f, .1f}, {0.0F, 1.0F}},
    {{1.0F, 1.0F, 0.0F}, {.1f, .8f, .1f}, {1.0F, 1.0F}},
    {{1.0F, 0.0F, 0.0F}, {.1f, .8f, .1f}, {1.0F, 0.0F}},
};

vec3 triangle_colors[4] = vec3[](
    vec3(0.0, 0.0, 0.0),
    vec3(1.0, 0.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(0.0, 1.0, 0.0)
);

uint triangle_indices[] = {
    0, 1, 2,
    0, 2, 3,

    4, 5, 6,
    4, 6, 7,

    8, 9, 10,
    8, 10, 11,

    12, 13, 14,
    12, 14, 15,

    16, 17, 18,
    16, 18, 19,

    20, 21, 22,
    20, 22, 23
};

void main() {
    vec3 vert_color = triangle_vertices[triangle_indices[gl_VertexIndex]].color;
    vec2 vert_texcoord = triangle_vertices[triangle_indices[gl_VertexIndex]].texcoord;
    vec3 vert_position = triangle_vertices[triangle_indices[gl_VertexIndex]].position * (vec3(48, 12, 112) / 2.0F);

    gl_Position = world_to_clip * vec4(vert_position, 1.0);
    out_frag_color = vert_color;
    out_vert_position = vert_position;
    out_vert_texcoord = vert_texcoord;
    out_vert_direction = (vert_position - camera_position);
}