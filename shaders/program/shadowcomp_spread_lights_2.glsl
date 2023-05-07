#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(4, 2, 4);

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

void main() {
	float scores[64];
	int localLightCount = readVolumePointer(ivec3(gl_GlobalInvocationID.xyz), 4) - 1;
	if (localLightCount > 0) {
		vec3 midPos = ((gl_GlobalInvocationID.xyz + 0.5) * 4.0 - pointerGridSize / 2.0) * POINTER_VOLUME_RES;
		int lightStripLoc = readVolumePointer(ivec3(gl_GlobalInvocationID.xyz), 5) + 1;
		for (int k = 1; k < localLightCount; k++) {
			int thisPointer = readLightPointer(lightStripLoc + k);
			light_t thisLight = lights[thisPointer];
			float score = (thisLight.brightnessMat >> 16) - length(midPos - thisLight.pos);
			int j;
			for (j = 0; j < min(k, 64) && scores[j] > score; j++);
			for (int i = min(k, 63); i > j; i--) {
				writeLightPointer(lightStripLoc + i, readLightPointer(lightStripLoc + i - 1));
				scores[i] = scores[i-1];
			}
			if (j < 64) {
				writeLightPointer(lightStripLoc + j, thisPointer);
				scores[j] = score;
			}
		}
	}
}