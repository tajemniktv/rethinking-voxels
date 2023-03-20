#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(64, 32, 64);

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#include "/lib/vx/SSBOs.glsl"
#include "/lib/materials/shadowchecks_precise.glsl"

void main() {
	int triCountHere = triPointerVolume[0][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
	int sorted[LOCAL_MAX_TRIS];
	int mats[LOCAL_MAX_TRIS];
	int lightCombine[50][64];
	for (int i = 0; i < 64; i++) lightCombine[0][i] = 0;
	int nLights = 0;
	for (int i = 0; i < triCountHere; i++) {
		int thisTriId = triPointerVolume[i][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
		tri_t thisTri = tris[thisTriId];
		int mat = int(thisTri.matBools % 65536);
		mats[i] = mat;
		bool emissive = getEmissive(mat);
		if (emissive) {
			bool newLight = true;
			for (int j = 0; j < i; j++) {
				if (mats[j] == mat) {
					int otherTriId = triPointerVolume[j][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
					tri_t otherTri = tris[otherTriId];
					bool close = false;
					for (int k = 0; k < 3; k++) {
						for (int l = 0; l < 3; l++) {
							if (length(thisTri.pos[k] - otherTri.pos[l]) < 0.5) {
								close = true;
								break;
							}
						}
						if (close) break;
					}
					if (close) {
						if (newLight) {
							sorted[i] = sorted[j];
							newLight = false;
						} else {
							bool alreadyCombined = false;
							for (int k = 1; k <= lightCombine[0][sorted[j]]; k++) {
								if (lightCombine[k][sorted[j]] == sorted[i]) alreadyCombined = true;
							}
							if (!alreadyCombined) {
								lightCombine[0][sorted[j]]++;
								lightCombine[0][sorted[i]]++;
								lightCombine[lightCombine[0][sorted[j]]][sorted[j]] = sorted[i];
								lightCombine[lightCombine[0][sorted[i]]][sorted[i]] = sorted[j];
							}
						}
					}
				}
			}
			if (newLight) {
				nLights++;
				sorted[i] = nLights;
			}
		} else sorted[i] = -1;
	}
	for (int i = 0; i < 64; i++) {
		int actualClump = i;
		for (int j = 1; j <= lightCombine[0][i]; j++) actualClump = min(actualClump, lightCombine[j][i]);
		lightCombine[0][i] = actualClump;
	}
	int lightClumps[LOCAL_MAX_TRIS][64];
	for (int i = 0; i < 64; i++) lightClumps[0][i] = 0;
	for (int i = 0; i < triCountHere; i++) {
		int clump = lightCombine[0][sorted[i]];
		lightClumps[++lightClumps[0][clump]][clump] = i;
	}
	for (int i = 0; i < 64; i++) {
		if (lightClumps[0][i] == 0) continue;
		int mat = 0;
		vec3 lower = vec3( 1000000000.0);
		vec3 upper = vec3(-1000000000.0);
		for (int j = 1; j <= lightClumps[0][i]; j++) {
			int thisTriId = triPointerVolume[lightClumps[j][i]][gl_GlobalInvocationID.x][gl_GlobalInvocationID.y][gl_GlobalInvocationID.z];
			tri_t thisTri = tris[thisTriId];
			if (mat == 0) mat = int(thisTri.matBools % 65536);
			vec3 thisLower = vec3( 1000000000.0);
			vec3 thisUpper = vec3(-1000000000.0);
			for (int k = 0; k < 3; k++) {
				thisLower = min(thisLower, thisTri.pos[k]);
				thisUpper = max(thisUpper, thisTri.pos[k]);
			}
			vec3 avg = 0.5 * (thisLower + thisUpper);
			lower = min(lower, avg);
			upper = max(upper, avg);
		}
		vec3 avg = 0.5 * (upper + lower);
		vec3 size = 0.5 * (upper - lower);
		vec3 lightCol = getLightCol(mat);
		int lightLevel = getLightLevel(mat);
		light_t thisLight;
		thisLight.pos = avg;
		thisLight.size = size;
		thisLight.packedColor = uint(255 * lightCol.x + 0.5) + (uint(255 * lightCol.x + 0.5) << 8) + (uint(255 * lightCol.x + 0.5) << 16);
		thisLight.brightnessMat = uint(mat + (lightLevel << 16));
		int globalLightId = atomicAdd(numLights, 1);
		lights[globalLightId] = thisLight;
		for (int x = -lightLevel/2 - 1; x <= lightLevel/2 + 1; x++) {
			int xCoord = x + int(gl_GlobalInvocationID.x);
			if (xCoord >= 0 && xCoord < 64) {
				for (int y = -lightLevel/2 - 1; y <= lightLevel/2 + 1; y++) {
					int yCoord = y + int(gl_GlobalInvocationID.y);
					if (yCoord >= 0 && yCoord < 32) {
						for (int z = -lightLevel/2 - 1; z <= lightLevel/2 + 1; z++) {
							int zCoord = z + int(gl_GlobalInvocationID.z);
							if (zCoord >= 0 && zCoord < 64) {
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