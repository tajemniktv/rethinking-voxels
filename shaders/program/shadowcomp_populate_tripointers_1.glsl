layout(local_size_x = 1) in;

const ivec3 workGroups = ivec3(64, 32, 64);

#include "/lib/vx/SSBOs.glsl"

ivec3 intWorkGroupId = abs(ivec3(gl_WorkGroupID));
int threadNum = intWorkGroupId.x + workGroups.x * intWorkGroupId.y + workGroups.x * workGroups.y * intWorkGroupId.z;
const int threadCount = workGroups.x * workGroups.y * workGroups.z;
const int strideSize = max(MAX_TRIS / threadCount, 1);

void main() {
	for (int i = strideSize * threadNum; i < strideSize * (threadNum + 1); i++) {
		if (i >= numFaces) break;
		atomicAdd(tris[0].bvhParent, 1);
		tri_t thisTri = tris[i];
		ivec3 pointerGridCoord = ivec3(
			thisTri.bvhParent % pointerGridSize.x,
			thisTri.bvhParent / pointerGridSize.x % pointerGridSize.y,
			thisTri.bvhParent / (pointerGridSize.x * pointerGridSize.y) % pointerGridSize.z
		);
		int globalAddr = PointerVolume[1][pointerGridCoord.x][pointerGridCoord.y][pointerGridCoord.z];
		int localAddr = atomicAdd(bvhLeaves[globalAddr], 1);
		if (localAddr <= PointerVolume[0][pointerGridCoord.x][pointerGridCoord.y][pointerGridCoord.z]) {
			atomicExchange(bvhLeaves[globalAddr + localAddr], i);
		}
	}
}