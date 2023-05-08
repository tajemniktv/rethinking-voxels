#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(64, 32, 64);

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#ifndef ACCURATE_RT

uniform int frameCounter;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/voxelReading.glsl"

void main() {
	const vec3 defaultLightSize = 0.5 * vec3(BLOCKLIGHT_SOURCE_SIZE);
	const int intpointerVolumeRes = int(POINTER_VOLUME_RES + 0.001);
	int nLights = 0;
	for (int x0 = 0; x0 < intpointerVolumeRes; x0++) {
		for (int y0 = 0; y0 < intpointerVolumeRes; y0++) {
			for (int z0 = 0; z0 < intpointerVolumeRes; z0++) {
				ivec3 blockCoord = intpointerVolumeRes * ivec3(gl_WorkGroupID) + ivec3(x0, y0, z0);
				if (readVoxelVolume(blockCoord, 0).x == 0) {
					writeVoxelVolume(blockCoord, 1, uvec4(0));
					continue;
				}
				writeVoxelVolume(blockCoord, 0, uvec4(0));
				vxData thisVoxelData = readVxMap(blockCoord);
				if (thisVoxelData.emissive) {
					light_t thisLight;
					thisLight.pos = blockCoord - POINTER_VOLUME_RES * pointerGridSize / 2.0;
					if (thisVoxelData.cuboid) {
						thisLight.pos += 0.5 * (thisVoxelData.upper + thisVoxelData.lower);
						#ifdef CORRECT_CUBOID_OFFSETS
							thisLight.size = 0.5 * (thisVoxelData.upper - thisVoxelData.lower);
						#else
							thisLight.size = defaultLightSize;
						#endif
					} else {
						thisLight.pos += thisVoxelData.midcoord;
						#ifdef CORRECT_CUBOID_OFFSETS
							thisLight.size = thisVoxelData.full ? vec3(0.5) : defaultLightSize;
						#else
							thisLight.size = defaultLightSize;
						#endif
					}
					thisLight.packedColor = int(thisVoxelData.lightcol.x * 255.9) + (int(thisVoxelData.lightcol.y * 255.9) << 8) + (int(thisVoxelData.lightcol.z * 255.9) << 16);
					thisLight.brightnessMat = thisVoxelData.mat + (thisVoxelData.lightlevel << 16);
					int globalLightId = atomicAdd(numLights, 1);
					if (globalLightId < MAX_LIGHTS) {
						lights[globalLightId] = thisLight;
						//pointerVolume[5 + nLights][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = globalLightId;
						nLights++;
					} else break;
					ivec3 coords = ivec3(thisLight.pos / POINTER_VOLUME_RES + pointerGridSize / 2) / 4;
					ivec3 lowerBound = max(coords - thisVoxelData.lightlevel / int(4.01 * POINTER_VOLUME_RES) - 1, ivec3(0));
					ivec3 upperBound = min(coords + thisVoxelData.lightlevel / int(4.01 * POINTER_VOLUME_RES) + 1, pointerGridSize / 4);
					for (int x = lowerBound.x; x <= upperBound.x; x++) {
						for (int y = lowerBound.y; y <= upperBound.y; y++) {
							for (int z = lowerBound.z; z <= upperBound.z; z++) {
								incrementVolumePointer(ivec3(x, y, z), 4);
							}
						}
					}
				}
			}
		}
	}
	//pointerVolume[4][gl_WorkGroupID.x][gl_WorkGroupID.y][gl_WorkGroupID.z] = nLights;
}
#else
void main() {}
#endif