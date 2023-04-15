#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(16, 8, 16);

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

shared int lightCount = 0;
shared int[128] lightPointers;

void main() {
	int nLights = PointerVolume[4][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
	for (int i = 0; i < nLights; i++) {
		int sharedLightId = atomicAdd(lightCount, 1);
		if (sharedLightId < 128) {
			lightPointers[sharedLightId] = PointerVolume[5 + i][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
		}
	}
	vec3 pos = POINTER_VOLUME_RES * (0.5 + gl_GlobalInvocationID - pointerGridSize / 2);
	groupMemoryBarrier();
	if (gl_LocalInvocationID == uvec3(0)) {
		lightCount = min(lightCount, 128);
		for (int i = 0; i < lightCount; i++) {
			PointerVolume[5 + i][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = lightPointers[i];
		}
		PointerVolume[4][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = lightCount;
	} else if (gl_LocalInvocationID.yz == uvec2(0)) {
		PointerVolume[4][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = 0;
	}
}