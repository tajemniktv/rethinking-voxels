#include "/lib/common.glsl"

layout(local_size_x = 1) in;

const ivec3 workGroups = ivec3(64, 32, 64);

#ifdef ACCURATE_RT
#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

ivec3 intWorkGroupId = abs(ivec3(gl_WorkGroupID));
int threadNum = intWorkGroupId.x + workGroups.x * intWorkGroupId.y + workGroups.x * workGroups.y * intWorkGroupId.z;
const int threadCount = workGroups.x * workGroups.y * workGroups.z;
const int strideSize = max(MAX_TRIS / threadCount, 1);

void main() {
	for (int i = strideSize * threadNum; i < strideSize * (threadNum + 1); i++) {
		if (i >= numFaces) break;
		tri_t thisTri = tris[i];
		ivec3 pointerGridCoord = ivec3(
			thisTri.bvhParent % 512,
			thisTri.bvhParent / 512 % 512,
			thisTri.bvhParent / 262144 % 512
		) / int(POINTER_VOLUME_RES + 0.1);
		int globalAddr = readVolumePointer(pointerGridCoord, 1);//pointerVolume[1][pointerGridCoord.x][pointerGridCoord.y][pointerGridCoord.z];
		if (globalAddr < maxStripIndex) {
			int localAddr = incrementTriPointer(globalAddr);//atomicAdd(triPointerStrip[globalAddr], 1);
			if (localAddr <= readVolumePointer(pointerGridCoord, 0) /*pointerVolume[0][pointerGridCoord.x][pointerGridCoord.y][pointerGridCoord.z]*/ && globalAddr < maxStripIndex) {
				writeTriPointer(globalAddr + localAddr, i);//triPointerStrip[globalAddr + localAddr] = i;
			}
		}
	}
}
#else
void main() {}
#endif