layout(local_size_x = 64) in;

const ivec3 workGroups = ivec3(1, 1, 1);

#include "/lib/vx/SSBOs.glsl"

shared int totalCounts[64];

void main() {
	int thisTotalCount = 0;
	int x = int(gl_LocalInvocationID.x);
	for (int y = 0; y < pointerGridSize.y; y++) {
		for (int z = 0; z < pointerGridSize.z; z++) {
			PointerVolume[1][x][y][z] = thisTotalCount;
			thisTotalCount += PointerVolume[0][x][y][z] + 1;
		}
	}
	totalCounts[x] = thisTotalCount;
	groupMemoryBarrier();
	if (x == 0) {
		for (int x0 = 1; x0 < pointerGridSize.x; x0++) {
			totalCounts[x0] += totalCounts[x0-1];
		}
	}
	groupMemoryBarrier();
	totalCounts[x] -= thisTotalCount;
	for (int y = 0; y < pointerGridSize.y; y++) {
		for (int z = 0; z < pointerGridSize.z; z++) {
			bvhLeaves[totalCounts[x]] = 1;
			PointerVolume[1][x][y][z] = totalCounts[x];
			totalCounts[x] += PointerVolume[0][x][y][z] + 1;
		}
	}
}
