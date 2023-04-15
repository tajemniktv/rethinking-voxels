#include "/lib/common.glsl"

const ivec3 workGroups = ivec3(16, 8, 16);

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

#define WRITE_TO_SSBOS
#include "/lib/vx/SSBOs.glsl"

void main() {
	ivec3 coords0 = 4 * ivec3(gl_WorkGroupID);
	int nLights = PointerVolume[4][coords0.x][coords0.y][coords0.z];
	for (int i = 0; i < nLights; i++) {
		int pointer = PointerVolume[5 + i][coords0.x][coords0.y][coords0.z];
		int lightlevel = lights[pointer].brightnessMat >> 16;
		int range = int(lightlevel / (4 * POINTER_VOLUME_RES) + 1.5);
		ivec2[3] bounds;
		for (int j = 0; j < 3; j++) {
			bounds[j].x = max(-range, -int(gl_WorkGroupID[i]));
			bounds[j].y = min(range + 1, pointerGridSize[i] / 4 - int(gl_WorkGroupID[i]));
		}
		for (int x = bounds[0].x; x < bounds[0].y; x++) {
			int xid = 4 * (int(gl_WorkGroupID.x) + x) + 1;
			for (int y = bounds[1].x; y < bounds[1].y; y++) {
				if (length(vec2(x, y)) > range) continue;
				int yid = 4 * (int(gl_WorkGroupID.y) + y);
				for (int z = bounds[2].x; z < bounds[2].y; z++) {
					if (length(vec3(x, y, z)) > range) continue;
					int zid = 4 * (int(gl_WorkGroupID.z) + z);
					int localLightId = atomicAdd(PointerVolume[4][xid][yid][zid], 1);
					if (localLightId < 64) PointerVolume[5 + localLightId][xid][yid][zid] = pointer;
					else {
						localLightId = atomicAdd(PointerVolume[4][xid + 1][yid][zid], 1);
						if (localLightId < 64) PointerVolume[5 + localLightId][xid + 1][yid][zid] = pointer;
						else {
							localLightId = atomicAdd(PointerVolume[4][xid + 2][yid][zid], 1);
							if (localLightId < 64) PointerVolume[5 + localLightId][xid + 2][yid][zid] = pointer;
						}
					}
				}
			}
		}
	}
}