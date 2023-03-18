#ifndef SSBO
#define SSBO
#define LOCAL_MAX_TRIS 512
#define MAX_TRIS 524288
#define POINTER_VOLUME_RES 2.0
struct entry_t {
	uint matBools;
	uvec3 texCoord;
	mat3 pos;
};
layout(std430, binding = 0) buffer voxelData {
	int numFaces;
	entry_t entries[];
};

layout(std430, binding = 1) buffer voxelPointers {
	int pointerVolume[][64][32][64];
};
#endif