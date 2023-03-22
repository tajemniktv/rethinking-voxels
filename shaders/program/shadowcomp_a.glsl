const ivec3 workGroups = ivec3(64, 32, 64);

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "/lib/common.glsl"

#include "/lib/vx/SSBOs.glsl"
#include "/lib/materials/shadowchecks_precise.glsl"

#define LOCAL_MAX_LIGHTCOUNT 8
void main() {
	int triCountHere = triPointerVolume[0][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
	triCountHere = min(triCountHere, LOCAL_MAX_TRIS);
	int mats[LOCAL_MAX_LIGHTCOUNT];
	int locs[LOCAL_MAX_LIGHTCOUNT];
	int nLights = 0;
	int lightClumps[LOCAL_MAX_TRIS][LOCAL_MAX_LIGHTCOUNT];
	int lightClumpTriCounts[LOCAL_MAX_LIGHTCOUNT];
	for (int i = 0; i < LOCAL_MAX_LIGHTCOUNT; i++) {
		mats[i] = -1;
		locs[i] = -1;
		lightClumpTriCounts[i] = 0;
	}
	for (int i = 0; i < triCountHere; i++) {
		int thisTriId = triPointerVolume[i][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
		tri_t thisTri = tris[thisTriId];
		int mat = int(thisTri.matBools % 65536);
		bool emissive = getEmissive(mat);
		if (emissive) {
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
			vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
			avgPos -= 0.05 * normalize(cnormal);
			vec3 localPos = avgPos / POINTER_VOLUME_RES - vec3(gl_WorkGroupID) + vec3(32, 16, 32);
			int loc = int(localPos.x > 0.5) + (int(localPos.y > 0.5) << 1) + (int(localPos.z > 0.5) << 2);
			bool newLight = true;
			for (int j = 0; j < nLights; j++) {
				if (mats[j] == mat && locs[j] == loc) {
					newLight = false;
					lightClumps[lightClumpTriCounts[j]][j] = i;
					lightClumpTriCounts[j]++;
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
	for (int i = 0; i < nLights; i++) {
		if (lightClumps[0][i] == 0) break;
		int mat = mats[i];
		vec3 lower = vec3( 1000000000.0);
		vec3 upper = vec3(-1000000000.0);
		float area = 0;
		bool avgBound = false;
		if ( // torches have way too large geometry, so a hack is needed to correctly detect their size
			mat == 10496 || // torch
			mat == 10497 ||
			mat == 10528 || // soul torch
			mat == 10529 ||
			mat == 12604 || // lit redstone torch
			mat == 12605
		) avgBound = true;
		for (int j = 0; j < lightClumpTriCounts[i]; j++) {
			int thisTriId = triPointerVolume[lightClumps[j][i]][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z];
			tri_t thisTri = tris[thisTriId];
			vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
			area += length(cnormal);
			vec3 thisLower = vec3( 1000000000.0);
			vec3 thisUpper = vec3(-1000000000.0);
			for (int k = 0; k < 3; k++) {
				thisLower = min(thisLower, thisTri.pos[k]);
				thisUpper = max(thisUpper, thisTri.pos[k]);
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
		vec3 avg = 0.5 * (upper + lower);
		vec3 size = 0.5 * (upper - lower);
		vec3 lightCol = getLightCol(mat);
		//lightCol *= area * 5;
		int lightLevel = getLightLevel(mat);
		light_t thisLight;
		thisLight.pos = avg;
		thisLight.size = size;
		thisLight.packedColor = int(255 * lightCol.x + 0.5) + (int(255 * lightCol.y + 0.5) << 8) + (int(255 * lightCol.z + 0.5) << 16);
		thisLight.brightnessMat = mat + (lightLevel << 16);
		int globalLightId = atomicAdd(numLights, 1);
		lights[globalLightId] = thisLight;

		for (int x = -lightLevel/2 - 1; x <= lightLevel/2 + 1; x++) {
			int xCoord = x + int(gl_WorkGroupID.x);
			if (xCoord >= 0 && xCoord < 64) {
				for (int y = -lightLevel/2 - 1; y <= lightLevel/2 + 1; y++) {
					int yCoord = y + int(gl_WorkGroupID.y);
					if (yCoord >= 0 && yCoord < 32 && length(vec2(x, y)) < lightLevel/2 + 1) {
						for (int z = -lightLevel/2 - 1; z <= lightLevel/2 + 1; z++) {
							int zCoord = z + int(gl_WorkGroupID.z);
							if (zCoord >= 0 && zCoord < 64 && length(vec3(x, y, z)) < lightLevel/2 + 1) {
								int localLightId = atomicAdd(lightPointerVolume[0][xCoord][yCoord][zCoord], 1);
								if (localLightId < 64) lightPointerVolume[localLightId + 1][xCoord][yCoord][zCoord] = globalLightId;
							}
						}
					}
				}
			}
		}
	}
}