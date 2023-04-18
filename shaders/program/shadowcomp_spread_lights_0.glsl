#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(16, 8, 16);

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

uniform int frameCounter;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

shared int lightCount = 0;
shared int[64] lightPointers;

void main() {
	int nLights = pointerVolume[4][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
	for (int i = 0; i < nLights; i++) {
		int sharedLightId = atomicAdd(lightCount, 1);
		if (sharedLightId < 64) {
			lightPointers[sharedLightId] = pointerVolume[5 + i][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
		}
	}
	vec3 pos = POINTER_VOLUME_RES * (0.5 + gl_GlobalInvocationID - pointerGridSize / 2);
	groupMemoryBarrier();
	if (gl_LocalInvocationID == uvec3(0)) {
		lightCount = min(lightCount, 64);
		for (int i = 0; i < lightCount; i++) {
			pointerVolume[5 + i][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = lightPointers[i];
		}
		pointerVolume[4][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = lightCount;
	} else if (gl_LocalInvocationID.yz == uvec2(0)) {
		pointerVolume[4][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = 0;
	}
}