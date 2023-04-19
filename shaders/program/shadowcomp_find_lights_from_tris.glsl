#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(64, 32, 64);

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#ifdef ACCURATE_RT

uniform int frameCounter;
uniform sampler2D colortex15;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"
#include "/lib/materials/shadowchecks_precise.glsl"

#define LOCAL_MAX_LIGHTCOUNT 8
void main() {
	vec3 thisVoxelLower = vec3(gl_WorkGroupID - pointerGridSize * 0.5);
	vec3 thisVoxelUpper = vec3(gl_WorkGroupID - pointerGridSize * 0.5 + 1);
	int triCountHere = pointerVolume[0][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
	int triStripStart = pointerVolume[1][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
	triCountHere = min(triCountHere, maxStripIndex - triStripStart);
	int mats[LOCAL_MAX_LIGHTCOUNT];
	ivec3 locs[LOCAL_MAX_LIGHTCOUNT];
	int nLights = 0;
	int lightClumps[LOCAL_MAX_TRIS/2][LOCAL_MAX_LIGHTCOUNT];
	int lightClumpTriCounts[LOCAL_MAX_LIGHTCOUNT];
	vec3 lowerBound = vec3(100000);
	vec3 upperBound = vec3(-100000);
	for (int i = 0; i < LOCAL_MAX_LIGHTCOUNT; i++) {
		mats[i] = -1;
		locs[i] = ivec3(-1);
		lightClumpTriCounts[i] = 0;
	}
	for (int i = 1; i <= triCountHere; i++) {
		int thisTriId = triPointerStrip[triStripStart + i];
		tri_t thisTri = tris[thisTriId];
		vec3 lower0 = min(min(
			thisTri.pos[0],
			thisTri.pos[1]),
			thisTri.pos[2]
		);
		vec3 upper0 = max(max(
			thisTri.pos[0],
			thisTri.pos[1]),
			thisTri.pos[2]
		);
		lowerBound = min(lower0, lowerBound);
		upperBound = max(upper0, upperBound);
		int mat = int(thisTri.matBools % 65536);
		bool emissive = getEmissive(mat);
		if (emissive) {
			vec3 avgPos = 0.5 * (lower0 + upper0);
			vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
			//avgPos -= 0.01 * normalize(cnormal);
			vec3 localPos = avgPos / POINTER_VOLUME_RES - thisVoxelLower;
			ivec3 loc = ivec3(greaterThan(localPos, vec3(0.5)));
			bool newLight = true;
			for (int j = 0; j < nLights; j++) {
				vec3 dpos = abs(localPos - locs[j] * 0.5 - 0.25);
				if (mats[j] == mat && max(max(dpos.x, dpos.y), dpos.z) < 0.27) {
					newLight = false;
					lightClumps[lightClumpTriCounts[j]][j] = thisTriId;
					if (lightClumpTriCounts[j] < LOCAL_MAX_TRIS / 2 - 1) lightClumpTriCounts[j]++;
				}
			}
			if (newLight && nLights < LOCAL_MAX_LIGHTCOUNT) {
				mats[nLights] = mat;
				locs[nLights] = loc;
				lightClumps[0][nLights] = thisTriId;
				lightClumpTriCounts[nLights] = 1;
				nLights++;
			}
		}
	}
	lowerBound = lowerBound / POINTER_VOLUME_RES - vec3(gl_WorkGroupID) + pointerGridSize / 2;
	upperBound = upperBound / POINTER_VOLUME_RES - vec3(gl_WorkGroupID) + pointerGridSize / 2;
	ivec3 lowerBoundQuantized = ivec3(clamp(lowerBound, vec3(0), vec3(1)) * 31.0 - 0.9);
	ivec3 upperBoundQuantized = ivec3(clamp(upperBound, vec3(0), vec3(1)) * 31.0 + 0.9);
	int packedBounds =
		lowerBoundQuantized.x +
		(lowerBoundQuantized.y << 5) +
		(lowerBoundQuantized.z << 10) +
		(upperBoundQuantized.x << 15) +
		(upperBoundQuantized.y << 20) +
		(upperBoundQuantized.z << 25);
	pointerVolume[2][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = packedBounds;
	for (int i = 0; i < nLights; i++) {
		int mat = mats[i];
		vec3 lower = vec3( 10000.0);
		vec3 upper = vec3(-10000.0);
		float area = 0;
		bool detectLightCol = false;
		bool avgBound = false;
		vec3 lightCol = getLightCol(mat);
		float lightColSamples = 0;
		if (lightCol == vec3(0)) {
			 detectLightCol = true;
		}
		if ( // torches have way too large geometry, so a hack is needed to correctly detect their size
			mat == 10496 || // torch
			mat == 10497 ||
			mat == 10528 || // soul torch
			mat == 10529 ||
			mat == 12604 || // lit redstone torch
			mat == 12605
		) avgBound = true;
		for (int j = 0; j < lightClumpTriCounts[i]; j++) {
			tri_t thisTri = tris[lightClumps[j][i]];
			vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
			area += length(cnormal);
			vec3 thisLower = min(min(
				thisTri.pos[0],
				thisTri.pos[1]),
				thisTri.pos[2]
			);
			vec3 thisUpper = max(max(
				thisTri.pos[0],
				thisTri.pos[1]),
				thisTri.pos[2]
			);
			//thisLower = clamp(thisLower, thisVoxelLower * POINTER_VOLUME_RES, thisVoxelUpper * POINTER_VOLUME_RES);
			//thisUpper = clamp(thisUpper, thisVoxelLower * POINTER_VOLUME_RES, thisVoxelUpper * POINTER_VOLUME_RES);
			if (avgBound) {
				vec3 avg0 = 0.5 * (thisLower + thisUpper);
				lower = min(lower, avg0);
				upper = max(upper, avg0);
			} else {
				lower = min(lower, thisLower);
				upper = max(upper, thisUpper);
			}
			if (detectLightCol) {
				ivec2 texCoord = (
					ivec2(thisTri.texCoord[0]%65536, thisTri.texCoord[0]/65536) +
					ivec2(thisTri.texCoord[1]%65536, thisTri.texCoord[1]/65536) +
					ivec2(thisTri.texCoord[2]%65536, thisTri.texCoord[2]/65536)
				) / 3;
				vec4 lightCol0 = texelFetch(colortex15, texCoord, 0);
				lightCol += lightCol0.rgb * lightCol0.a;
				lightColSamples += lightCol0.a;
			}
		}
		if (detectLightCol) {
			lightCol /= lightColSamples;
			lightCol = 0.97 * lightCol + 0.03;
		}
		vec3 avg = 0.5 * (upper + lower);
		#ifdef CORRECT_CUBOID_OFFSETS
			vec3 size = max(0.5 * (upper - lower), vec3(0.01));
		#else
			vec3 size = vec3(BLOCKLIGHT_SOURCE_SIZE);
		#endif
		int lightLevel = getLightLevel(mat);
		light_t thisLight;
		thisLight.pos = avg;
		thisLight.size = size;
		thisLight.packedColor = int(255 * lightCol.x + 0.5) + (int(255 * lightCol.y + 0.5) << 8) + (int(255 * lightCol.z + 0.5) << 16);
		thisLight.brightnessMat = mat + (lightLevel << 16);
		int globalLightId = atomicAdd(numLights, 1);
		if (globalLightId < MAX_LIGHTS) {
			lights[globalLightId] = thisLight;
			pointerVolume[5 + i][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = globalLightId;
		} else {
			nLights = i - 1;
			break;
		}
/*		for (int x = -lightLevel/2 - 1; x <= lightLevel/2 + 1; x++) {
			int xCoord = x + int(gl_WorkGroupID.x);
			if (xCoord >= 0 && xCoord < pointerGridSize.x) {
				for (int y = -lightLevel/2 - 1; y <= lightLevel/2 + 1; y++) {
					int yCoord = y + int(gl_WorkGroupID.y);
					if (yCoord >= 0 && yCoord < pointerGridSize.y && length(vec2(x, y)) < lightLevel/2 + 1) {
						for (int z = -lightLevel/2 - 1; z <= lightLevel/2 + 1; z++) {
							int zCoord = z + int(gl_WorkGroupID.z);
							if (zCoord >= 0 && zCoord < pointerGridSize.z && length(vec3(x, y, z)) < lightLevel/2 + 1) {
								int localLightId = atomicAdd(pointerVolume[4][xCoord][yCoord][zCoord], 1);
								if (localLightId < 64) pointerVolume[5 + localLightId][xCoord][yCoord][zCoord] = globalLightId;
							}
						}
					}
				}
			}
		}
*/	}
	pointerVolume[4][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = nLights;
}
#else
void main() {}
#endif