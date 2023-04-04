#ifndef SSBO
#define SSBO
#define LOCAL_MAX_TRIS 512
#define MAX_TRIS 524288
#define POINTER_VOLUME_RES 2.0
const ivec3 pointerGridSize = ivec3(64, 32, 64);
struct tri_t {
	uint matBools;
	int bvhParent;
	uvec3 texCoord;
	uvec3 vertexCol;
	mat3 pos;
};
#ifdef ACCURATE_RT
	layout(std430, binding = 0) buffer voxelData {
		int numFaces;
		tri_t tris[];
	};
#else
	layout(std430, binding = 0) buffer voxelData {
		int numFaces;
		uvec4 voxelVolume[][2 * pointerGridSize.x][2 * pointerGridSize.y][2 * pointerGridSize.z];
	};
#endif

layout(std430, binding = 1) buffer volumePointers {
	int PointerVolume[][pointerGridSize.x][pointerGridSize.y][pointerGridSize.z];
};

struct light_t {
	vec3 pos;
	vec3 size;
	int packedColor;
	int brightnessMat;
};
layout(std430, binding = 2) buffer lightData {
	int numLights;
	light_t lights[];
};

layout(std430, binding = 3) buffer pointerStrip {
	int bvhLeaves[];
};
#endif