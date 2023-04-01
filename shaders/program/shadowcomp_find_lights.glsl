#include "/lib/common.glsl"

#include "/lib/vx/SSBOs.glsl"
#include "/lib/materials/shadowchecks_precise.glsl"

uniform sampler2D colortex15;

const ivec3 workGroups = ivec3(64, 32, 64);

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#define LOCAL_MAX_LIGHTCOUNT 8
void main() {
	int triCountHere = PointerVolume[0][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
	int triStripStart = PointerVolume[1][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
	int mats[LOCAL_MAX_LIGHTCOUNT];
	int locs[LOCAL_MAX_LIGHTCOUNT];
	int nLights = 0;
	int lightClumps[LOCAL_MAX_TRIS][LOCAL_MAX_LIGHTCOUNT];
	int lightClumpTriCounts[LOCAL_MAX_LIGHTCOUNT];
	vec3 lowerBound = vec3(100000);
	vec3 upperBound = vec3(-100000);
	for (int i = 0; i < LOCAL_MAX_LIGHTCOUNT; i++) {
		mats[i] = -1;
		locs[i] = -1;
		lightClumpTriCounts[i] = 0;
	}
	for (int i = 1; i <= triCountHere; i++) {
		int thisTriId = bvhLeaves[triStripStart + i];
		tri_t thisTri = tris[thisTriId];
		vec3 lower = min(min(
			thisTri.pos[0],
			thisTri.pos[1]),
			thisTri.pos[2]
		);
		vec3 upper = max(max(
			thisTri.pos[0],
			thisTri.pos[1]),
			thisTri.pos[2]
		);
		lowerBound = min(lower, lowerBound);
		upperBound = max(upper, upperBound);
		int mat = int(thisTri.matBools % 65536);
		bool emissive = getEmissive(mat);
		if (emissive) {
			vec3 avgPos = 0.5 * (lower + upper);
			vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
			avgPos -= 0.05 * normalize(cnormal);
			vec3 localPos = avgPos / POINTER_VOLUME_RES - vec3(gl_WorkGroupID) + pointerGridSize / 2;
			int loc = int(localPos.x > 0.5) + (int(localPos.y > 0.5) << 1) + (int(localPos.z > 0.5) << 2);
			bool newLight = true;
			for (int j = 0; j < nLights; j++) {
				if (mats[j] == mat && locs[j] == loc) {
					newLight = false;
					lightClumps[lightClumpTriCounts[j]][j] = thisTriId;
					if (lightClumpTriCounts[j] < LOCAL_MAX_TRIS - 1) lightClumpTriCounts[j]++;
				}
			}
			if (newLight && nLights < LOCAL_MAX_LIGHTCOUNT) {
				mats[nLights] = mat;
				locs[nLights] = loc;
				lightClumps[0][nLights] = i;
				lightClumpTriCounts[nLights]++;
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
	PointerVolume[2][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = packedBounds;
	PointerVolume[4][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = nLights;
	light_t localLights[LOCAL_MAX_LIGHTCOUNT];
	int areas[LOCAL_MAX_LIGHTCOUNT];
	int lightIds[LOCAL_MAX_LIGHTCOUNT];
	for (int i = 0; i < nLights; i++) {
		int mat = mats[i];
		vec3 lower = vec3( 1000000000.0);
		vec3 upper = vec3(-1000000000.0);
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
			vec3 thisLower = vec3( 1000000000.0);
			vec3 thisUpper = vec3(-1000000000.0);
			for (int k = 0; k < 3; k++) {
				thisLower = min(thisLower, thisTri.pos[k]);
				thisUpper = max(thisUpper, thisTri.pos[k]);
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
			if (avgBound) {
				vec3 avg = 0.5 * (thisLower + thisUpper);
				lower = min(lower, avg);
				upper = max(upper, avg);
			} else {
				lower = min(lower, thisLower);
				upper = max(upper, thisUpper);
			}
		}
		if (detectLightCol) {
			lightCol /= lightColSamples;
			lightCol = 0.97 * lightCol + 0.03;
		}
		vec3 avg = 0.5 * (upper + lower);
		vec3 size = 0.5 * (upper - lower);
		int lightLevel = getLightLevel(mat);
		light_t thisLight;
		thisLight.pos = avg;
		thisLight.size = size;
		thisLight.packedColor = int(255 * lightCol.x + 0.5) + (int(255 * lightCol.y + 0.5) << 8) + (int(255 * lightCol.z + 0.5) << 16);
		thisLight.brightnessMat = mat + (lightLevel << 16);
		int globalLightId = atomicAdd(numLights, 1);
		lightIds[i] = globalLightId;
		lights[globalLightId] = thisLight;
		PointerVolume[5 + i][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = globalLightId;
		PointerVolume[5 + i + nLights][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = int(4096 * area);
		localLights[i] = thisLight;
		areas[i] = int(4096 * area);
	}
	memoryBarrierBuffer();
	for (int i = 0; i < nLights; i++) {
		int lightId = PointerVolume[5 + i][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
		vec3 lower = localLights[i].pos - localLights[i].size - 0.1;
		vec3 upper = localLights[i].pos + localLights[i].size + 0.1;
		vec3 newLower = lower;
		vec3 newUpper = upper;
		float size = dot(upper - lower, vec3(1));
		bool mergedAway = false;
		for (int x = -1; x <= 1; x++) {
			int xCoord = x + int(gl_WorkGroupID.x);
			if (xCoord >= 0 && xCoord < pointerGridSize.x) {
				for (int y = -1; y <= 1; y++) {
					int yCoord = y + int(gl_WorkGroupID.y);
					if (yCoord >= 0 && yCoord < pointerGridSize.y) {
						for (int z = -1; z <= 1; z++) {
							int zCoord = z + int(gl_WorkGroupID.z);
							if (zCoord >= 0 && zCoord < pointerGridSize.z) {
								int aroundNLights = PointerVolume[4][xCoord][yCoord][zCoord];
								for (int k = 0; k < aroundNLights; k++) {
									if (ivec4(x, y, z, k) == ivec4(0, 0, 0, i)) continue;
									int thisLightId = PointerVolume[5 + k][xCoord][yCoord][zCoord];
									int area = PointerVolume[5 + k + aroundNLights][xCoord][yCoord][zCoord];
									light_t thisLight = lights[thisLightId];
									if (thisLight.brightnessMat % 65535 == mats[i]) {
										vec3 aroundLower = thisLight.pos - 0.5 * thisLight.size - 0.05;
										vec3 aroundUpper = thisLight.pos + 0.5 * thisLight.size + 0.05;
										vec3 overlapLower = max(lower, aroundLower);
										vec3 overlapUpper = min(upper, aroundUpper);
										if (
											all(greaterThan(overlapUpper, overlapLower)) &&
											dot(overlapUpper - overlapLower, vec3(1)) > 0.5 * min(size, dot(aroundUpper - aroundLower, vec3(1)))
										) {
											if (area > areas[i]) {
												atomicExchange(lights[lightId].packedColor, -thisLightId);
												mergedAway = true;
											} else if (area < areas[i]) {
												newLower = min(newLower, aroundLower);
												newUpper = max(newUpper, aroundUpper);
											}

										}
									}
									if (mergedAway) break;
								}
								if (mergedAway) break;
							}
						}
						if (mergedAway) break;
					}
				}
				if (mergedAway) break;
			}
		}
	}
	memoryBarrierBuffer();
	// not using atomics!!
	for (int i = 0; i < nLights; i++) {
		int lightId = PointerVolume[5 + i][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
		if (lights[lightId].packedColor < 0) localLights[i].brightnessMat = -500000;
		vec3 lower = lights[lightId].pos;
		vec3 upper = lower;
		vec3 size = lights[lightId].size;
		lower -= size;
		upper += size;
		bool notThereYet = true;
		for (int k = 0; k < 20 && notThereYet; k++) {
			if (lights[lightId].packedColor > 0) notThereYet = false;
			else lightId = -lights[lightId].packedColor;
		}
		vec3 otherLower = lights[lightId].pos - lights[lightId].size;
		vec3 otherUpper = lights[lightId].pos + lights[lightId].size;
		otherLower = min(lower, otherLower);
		otherUpper = max(upper, otherUpper);
		lights[lightId].pos = 0.5 * (otherUpper + otherLower);
		lights[lightId].size = 0.5 * (otherUpper - otherLower);
	}
	memoryBarrierBuffer();
	for (int i = 0; i < nLights; i++) {
		int lightLevel = localLights[i].brightnessMat >> 16;
		for (int x = -lightLevel/2 - 1; x <= lightLevel/2 + 1; x++) {
			int xCoord = x + int(gl_WorkGroupID.x);
			if (xCoord >= 0 && xCoord < pointerGridSize.x) {
				for (int y = -lightLevel/2 - 1; y <= lightLevel/2 + 1; y++) {
					int yCoord = y + int(gl_WorkGroupID.y);
					if (yCoord >= 0 && yCoord < pointerGridSize.y && length(vec2(x, y)) < lightLevel/2 + 1) {
						for (int z = -lightLevel/2 - 1; z <= lightLevel/2 + 1; z++) {
							if (ivec3(x, y, z) == ivec3(0)) continue;
							int zCoord = z + int(gl_WorkGroupID.z);
							if (zCoord >= 0 && zCoord < pointerGridSize.z && length(vec3(x, y, z)) < lightLevel/2 + 1) {
								int localLightId = atomicAdd(PointerVolume[4][xCoord][yCoord][zCoord], 1);
								if (localLightId < 64) PointerVolume[5 + localLightId][xCoord][yCoord][zCoord] = lightIds[i];
							}
						}
					}
				}
			}
		}
	}

}