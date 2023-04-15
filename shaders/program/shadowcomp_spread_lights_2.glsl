#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(16, 8, 16);

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

shared int lightCount = 0;
shared int[192] lightPointers;

void main() {
	if (gl_LocalInvocationID == uvec3(0, 0, 0)) {
		for (int k = 1; k < 4; k++) {
			ivec3 coords = ivec3(gl_GlobalInvocationID) + ivec3(k, 0, 0);
			int lightCount0 = PointerVolume[4][coords.x][coords.y][coords.z];
			lightCount0 = min(lightCount0, 64);
			if (lightCount0 == 0) break;
			for (int i = 0; i < lightCount0; i++) {
				lightPointers[i + lightCount] = PointerVolume[5 + i][coords.x][coords.y][coords.z];
			}
			lightCount += lightCount0;
		}
	}
	groupMemoryBarrier();
	vec3 pos = POINTER_VOLUME_RES * (0.5 + gl_GlobalInvocationID - pointerGridSize / 2);
	int nLights = 0;
	for (int i = 0; i < lightCount; i++) {
		light_t thisLight = lights[lightPointers[i]];
		if (true || length(thisLight.pos - pos) < (thisLight.brightnessMat >> 16) + 0.79 * POINTER_VOLUME_RES) {
			PointerVolume[5 + nLights][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = lightPointers[i];
			nLights++;
		}
	}
	PointerVolume[4][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z] = nLights;
}