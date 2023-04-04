#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(64, 32, 64);

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#ifndef ACCURATE_RT
#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/voxelReading.glsl"

void main() {
	const vec3 defaultLightSize = 0.5 * vec3(BLOCKLIGHT_SOURCE_SIZE);
	const int intPointerVolumeRes = int(POINTER_VOLUME_RES + 0.001);
	for (int x0 = 0; x0 < intPointerVolumeRes; x0++) {
		for (int y0 = 0; y0 < intPointerVolumeRes; y0++) {
			for (int z0 = 0; z0 < intPointerVolumeRes; z0++) {
				ivec3 blockCoord = intPointerVolumeRes * ivec3(gl_WorkGroupID) + ivec3(x0, y0, z0);
				if (voxelVolume[0][blockCoord.x][blockCoord.y][blockCoord.z].x == 0) {
					voxelVolume[1][blockCoord.x][blockCoord.y][blockCoord.z] = uvec4(0);
					continue;
				}
				voxelVolume[0][blockCoord.x][blockCoord.y][blockCoord.z].x = 0;
				vxData thisVoxelData = readVxMap(blockCoord);
				if (thisVoxelData.emissive) {
					light_t thisLight;
					thisLight.pos = blockCoord - POINTER_VOLUME_RES * pointerGridSize / 2.0;
					if (thisVoxelData.cuboid) {
						thisLight.pos += 0.5 * (thisVoxelData.upper + thisVoxelData.lower);
						thisLight.size = 0.5 * (thisVoxelData.upper - thisVoxelData.lower);
					} else {
						thisLight.pos += thisVoxelData.midcoord;
						thisLight.size = defaultLightSize;
					}
					thisLight.packedColor = int(thisVoxelData.lightcol.x * 255.9) + (int(thisVoxelData.lightcol.y * 255.9) << 8) + (int(thisVoxelData.lightcol.z * 255.9) << 16);
					thisLight.brightnessMat = thisVoxelData.mat + (thisVoxelData.lightlevel << 16);
					int globalLightId = atomicAdd(numLights, 1);
					lights[globalLightId] = thisLight;
					for (int x = -thisVoxelData.lightlevel/2 - 1; x <= thisVoxelData.lightlevel/2 + 1; x++) {
						int xCoord = x + int(gl_WorkGroupID.x);
						if (xCoord >= 0 && xCoord < pointerGridSize.x) {
							for (int y = -thisVoxelData.lightlevel/2 - 1; y <= thisVoxelData.lightlevel/2 + 1; y++) {
								int yCoord = y + int(gl_WorkGroupID.y);
								if (yCoord >= 0 && yCoord < pointerGridSize.y && length(vec2(x, y)) < thisVoxelData.lightlevel/2 + 1) {
									for (int z = -thisVoxelData.lightlevel/2 - 1; z <= thisVoxelData.lightlevel/2 + 1; z++) {
										int zCoord = z + int(gl_WorkGroupID.z);
										if (zCoord >= 0 && zCoord < pointerGridSize.z && length(vec3(x, y, z)) < thisVoxelData.lightlevel/2 + 1) {
											int localLightId = atomicAdd(PointerVolume[4][xCoord][yCoord][zCoord], 1);
											if (localLightId < 64) PointerVolume[5 + localLightId][xCoord][yCoord][zCoord] = globalLightId;
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
#else
void main() {}
#endif