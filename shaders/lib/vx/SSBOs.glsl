#ifndef SSBO
#define SSBO
#define LOCAL_MAX_TRIS 512
#define MAX_TRIS 524288
#define POINTER_VOLUME_RES 2.0
struct tri_t {
	uint matBools;
	uvec3 texCoord;
	uvec3 vertexCol;
	mat3 pos;
};
layout(std430, binding = 0) buffer voxelData {
	int numFaces;
	tri_t tris[];
};

layout(std430, binding = 1) buffer triPointers {
	int triPointerVolume[][64][32][64];
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

layout(std430, binding = 3) buffer lightPointers {
	int lightPointerVolume[][32][16][32];
};

#endif