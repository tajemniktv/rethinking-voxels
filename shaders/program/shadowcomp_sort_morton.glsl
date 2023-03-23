layout(local_size_x = 1) in;

const ivec3 workGroups = ivec3(65536, 1, 1);

#include "/lib/common.glsl"
#include "/lib/vx/SSBOs.glsl"

void main() {
	sortingStuff[gl_WorkGroupID.x][2] = 0;
	int faceCount = numFaces;
	memoryBarrierBuffer();
	uint sorted = 0;
	for (int k = 0; k < (MAX_TRIS / 65536); k++) {
		int arrayIndex0 = (MAX_TRIS / 65536) * int(gl_WorkGroupID.x) + k;
		int arrayIndex1 = (tris[arrayIndex0].mortonCode >> 14);
		int localCount = atomicAdd(sortingStuff[arrayIndex1][2], 1);
		if (localCount < MAX_TRIS / 65536) {
			sortingStuff[(MAX_TRIS / 65536) * arrayIndex1 + localCount] = arrayIndex0;
			sorted += 1<<k;
		}
	}
}