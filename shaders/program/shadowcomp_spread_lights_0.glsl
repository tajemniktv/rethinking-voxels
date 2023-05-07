#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(1, 1, 1);

layout(local_size_x = 16) in;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

shared int totalCounts[pointerGridSize.x / 4];

void main() {
	int thisTotalCount = 0;
	int x = int(gl_LocalInvocationID.x);
	for (int y = 0; y < pointerGridSize.y / 4; y++) {
		for (int z = 0; z < pointerGridSize.z / 4; z++) {
			thisTotalCount += readVolumePointer(ivec3(x, y, z), 4) + 1;//pointerVolume[0][x][y][z] + 1;
		}
	}
	totalCounts[x] = thisTotalCount;
	barrier();
	groupMemoryBarrier();
	if (x == 0) {
		for (int x0 = 1; x0 < pointerGridSize.x / 4; x0++) {
			totalCounts[x0] += totalCounts[x0-1];
		}
	}
	barrier();
	groupMemoryBarrier();
	thisTotalCount = totalCounts[x] - thisTotalCount;
	for (int y = 0; y < pointerGridSize.y / 4; y++) {
		for (int z = 0; z < pointerGridSize.z / 4; z++) {
			writeLightPointer(thisTotalCount, 1);//triPointerStrip[thisTotalCount] = 1;
			writeVolumePointer(ivec3(x, y, z), 5, thisTotalCount);//pointerVolume[1][x][y][z] = thisTotalCount;
			thisTotalCount += readVolumePointer(ivec3(x, y, z), 4) + 1;//pointerVolume[0][x][y][z] + 1;
		}
	}
}