#version 450

layout(push_constant) uniform ShaderUniforms {
    mat4 world_to_clip;
    vec3 camera_position;
};

layout(binding = 0) uniform sampler2DArray car_volume;

layout(location = 0) in vec3 in_vert_position;
layout(location = 1) in vec2 in_vert_texcoord;
layout(location = 2) in vec3 in_vert_direction;

layout(location = 0) out vec4 out_frag_color;

const int MAX_RAY_STEPS = 256;

void main() {
//    out_frag_color = texture(car_volume, vec3(in_vert_texcoord.x, in_vert_texcoord.y, 0));
//    return;

    vec3 rayDir = normalize(in_vert_direction);
    vec3 rayPos = camera_position;

    ivec3 mapPos = ivec3(floor(rayPos));
    ivec3 rayStep = ivec3(sign(rayDir));

    vec3 deltaDist = abs(vec3(length(rayDir)) / rayDir);
    vec3 sideDist = (sign(rayDir) * (vec3(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist;

    bvec3 mask = bvec3(false);
    vec4 color = vec4(0, 0, 0, 0);
    for (int i = 0; i < MAX_RAY_STEPS; i++) {
        if (all(greaterThanEqual(mapPos, ivec3(0))) && all(lessThanEqual(mapPos, ivec3(48, 12, 112)))) {
            float s = float(mapPos.x) / float(48);
            float u = float(mapPos.y) / float(12) * 12;
            float t = float(mapPos.z) / float(112);
            vec4 rgba = texture(car_volume, vec3(s, t, u));
            if (rgba.r < 1.0F) {
                color.rgb = vec3(1);
                if (mask.x) {
                    color.rgb = vec3(0.5);
                }
                if (mask.y) {
                    color.rgb = vec3(1.0);
                }
                if (mask.z) {
                    color.rgb = vec3(0.75);
                }
                break;
            }
        }

        mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
        sideDist += vec3(mask) * deltaDist;
        mapPos += ivec3(vec3(mask)) * rayStep;
    }
    out_frag_color = color;
}