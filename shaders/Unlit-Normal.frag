#version 450

#extension GL_EXT_debug_printf : enable

#define COUNT_VOXELS 8
#define COUNT_STEPS 32

layout(push_constant) uniform ShaderUniforms {
    mat4 world_to_clip;
    vec3 camera_position;
};

layout(binding = 0) uniform sampler2DArray car_volume;

uint volume[16] = uint[16](
    0x00000000,
    0x00000000,
    0x42427E00,
    0x007E4242,
    0x00004200,
    0x00420000,
    0x00004200,
    0x00420000,
    0x00004200,
    0x00420000,
    0x00004200,
    0x00420000,
    0x42427E00,
    0x007E4242,
    0x00000000,
    0x00000000
);

vec3 intersectAABB(vec3 rayPos, vec3 rayDir, vec3 aabbMin, vec3 aabbMax) {
    vec3 tMin = (aabbMin - rayPos) / rayDir;
    vec3 tMax = (aabbMax - rayPos) / rayDir;
    vec3 t1 = min(tMin, tMax);
    vec3 t2 = max(tMin, tMax);
    float tNear = max(max(t1.x, t1.y), t1.z);
    float tFar = min(min(t2.x, t2.y), t2.z);
    return vec3(tNear, tFar, tFar - tNear);
}

layout(location = 0) in vec3 in_frag_color;
layout(location = 1) in vec3 in_vert_position;
layout(location = 2) in vec2 in_vert_texcoord;
layout(location = 3) in vec3 in_vert_direction;

layout(location = 0) out vec4 out_frag_color;

const int MAX_RAY_STEPS = 64;

float sdSphere(vec3 p, float d) { return length(p) - d; }

float sdBox( vec3 p, vec3 b ) {
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) +
    length(max(d, 0.0));
}

//bool getVoxel(ivec3 c) {
//    vec3 p = vec3(c) + vec3(0.5);
//    float d = min(max(-sdSphere(p, 7.5), sdBox(p, vec3(6.0))), -sdSphere(p, 25.0));
//    return d < 0.0;
//}

bool getVoxel(ivec3 pos) {
    if (pos.x < 0 || pos.x >= COUNT_VOXELS) {
        return false;
    }
    if (pos.y < 0 || pos.y >= COUNT_VOXELS) {
        return false;
    }
    if (pos.z < 0 || pos.z >= COUNT_VOXELS) {
        return false;
    }
    uint i = pos.x + pos.y * COUNT_VOXELS + pos.z * COUNT_VOXELS * COUNT_VOXELS;
    return (volume[i / 32] & (1u << (i % 32))) != 0;
}

void main() {
    out_frag_color = texture(car_volume, vec3(in_vert_texcoord.x, in_vert_texcoord.y, 0));

//    vec3 rayDir = normalize(in_vert_direction);
//    vec3 rayPos = camera_position;
//
//    rayPos = rayPos + rayDir * (max(0, intersectAABB(rayPos, rayDir, vec3(0.0F), vec3(1.0F)).x));
//    rayPos = rayPos * COUNT_VOXELS;
//
//    ivec3 mapPos = ivec3(floor(rayPos));
//    vec3 deltaDist = abs(vec3(length(rayDir)) / rayDir);
//    ivec3 rayStep = ivec3(sign(rayDir));
//    vec3 sideDist = (sign(rayDir) * (vec3(mapPos) - rayPos) + (sign(rayDir) * 0.5) + 0.5) * deltaDist;
//
//    bvec3 mask;
//
//    vec4 color = vec4(0, 0, 0, 0);
//    for (int i = 0; i < MAX_RAY_STEPS; i++) {
//        if (getVoxel(mapPos)) {
//            if (mask.x) {
//                color.rgb = vec3(0.5);
//            }
//            if (mask.y) {
//                color.rgb = vec3(1.0);
//            }
//            if (mask.z) {
//                color.rgb = vec3(0.75);
//            }
//            break;
//        }
//
////        ivec3 c = mapPos;// - ivec3(fragVertexOrigin);
////        if (c.x >= 0 && c.y >= 0 && c.z >= 0 && c.x < COUNT_VOXELS && c.y < COUNT_VOXELS && c.z < COUNT_VOXELS) {
////            uint index = c.x + c.y * COUNT_VOXELS + c.z * COUNT_VOXELS * COUNT_VOXELS;
////            if ((volume[index / 32] & (1u << (index % 32))) != 0) {
////                out_frag_color = vec4(in_frag_color, 1.0F);
//////                NormalOutput = vec4(normalize(fragDirection), 1.0F);
//////                PositionOutput = vec4(fragOrigin, 1.0F);
////                return;
////            }
////        }
////        if (getVoxel(mapPos)) break;
//
//        mask = lessThanEqual(sideDist.xyz, min(sideDist.yzx, sideDist.zxy));
//        sideDist += vec3(mask) * deltaDist;
//        mapPos += ivec3(vec3(mask)) * rayStep;
//    }
//    out_frag_color = color;
}