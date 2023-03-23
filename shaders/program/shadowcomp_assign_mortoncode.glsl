layout(local_size_x = 1) in;

const ivec3 workGroups = ivec3(1024, 1, 1);

#include "/lib/common.glsl"
#include "/lib/vx/SSBOs.glsl"

int getMorton(ivec3 mortonVec) {
	int returnVal = 0;
	for (int i = 0; i < 3; i++) {
		int spread = 0;
		for (int j = 0; j < 10; j++) {
			spread += (mortonVec[i] & (1<<j)) << (2*j);
		}
		returnVal += spread << i;
	}
	return returnVal;
}

void main() {
	int remaining = min(512, int(numFaces - 512 * gl_WorkGroupID.x));
	for (int i = 0; i < remaining; i++) {
		int thisTriId = i + int(512 * gl_WorkGroupID.x);
		tri_t thisTri = tris[thisTriId];
		vec3 avgPos = 0.5 * (
			min(min(
				thisTri.pos[0],
				thisTri.pos[1]),
				thisTri.pos[2]
			) +
			max(max(
				thisTri.pos[0],
				thisTri.pos[1]),
				thisTri.pos[2]
			)
		);
		ivec3 mortonVec = ivec3((avgPos + 0.5 * POINTER_VOLUME_RES * pointerGridSize) * 8.0);
		tris[thisTriId].mortonCode = getMorton(mortonVec);
	}
}