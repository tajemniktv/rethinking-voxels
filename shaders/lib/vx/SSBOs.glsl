#ifndef SSBO
#define SSBO
#define LOCAL_MAX_TRIS 512
#define MAX_TRIS 524288
#define POINTER_VOLUME_RES 2.0
const ivec3 pointerGridSize = ivec3(64, 32, 64);
struct tri_t {
	uint matBools;
	int mortonCode;
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
	int lightPointerVolume[][64][32][64];
};

struct bvh_entry_t {
	vec3 lower;
	vec3 upper;
	int children[8];
	int childNum_isLeaf;
};
layout(std430, binding = 4) buffer bvh {
	int numBvhEntries;
	bvh_entry_t bvhEntries[];
};

layout(std430, binding = 5) buffer sortingBuffer {
	int pivot;
	int startBuffer;
	int sortingStuff[][3];
};
#endif