#include "/lib/common.glsl"

layout(local_size_x = 64) in;
//layout(local_size_x = 1) in;
const ivec3 workGroups = ivec3(1, 1, 1);

#ifdef ACCURATE_RT
#include "/lib/vx/SSBOs.glsl"

shared int totalCounts[pointerGridSize.x];

void main() {

	int thisTotalCount = 0;
	/*for (int x = 0; x < pointerGridSize.x; x++) {
		for (int y = 0; y < pointerGridSize.y; y++) {
			for (int z = 0; z < pointerGridSize.z; z++) {
				PointerVolume[1][x][y][z] = thisTotalCount;
				thisTotalCount += PointerVolume[0][x][y][z] + 1;
			}
		}
	}
	return;*/
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
#else
void main() {}
#endif