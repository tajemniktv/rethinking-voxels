#ifndef READING
#define READING

#include "/lib/vx/SSBOs.glsl"

struct vxData {
	vec3 lower;
	vec3 upper;
	vec3 midcoord;
	vec3 lightcol;
	ivec2 texelcoord;
	int spritesize;
	int mat;
	int lightlevel;
	int skylight;
	bool trace;
	bool full;
	bool cuboid;
	bool alphatest;
	bool emissive;
	bool crossmodel;
	bool connectsides;
	bool entity;
};

//read data from the voxel map (excluding flood fill data)
vxData readVxMap(ivec3 coords) {
	vxData data;
	#ifndef ACCURATE_RT
		int nonConstant1Index_nvidiaIsStupid = max(1, -coords.x);
		uvec4 packedData = voxelVolume[nonConstant1Index_nvidiaIsStupid][coords.x][coords.y][coords.z];
		if (packedData.x == 0) {
		#endif
		data.lightcol = vec3(0); // lightcol is gl_Color.rgb for anything that isn't a light source
		data.texelcoord = ivec2(-1);
		data.lower = vec3(0);
		data.upper = vec3(0);
		data.midcoord = vec3(0.5);
		data.mat = -1;
		data.full = false;
		data.cuboid = false;
		data.alphatest = false;
		data.trace = false;
		data.emissive = false;
		data.crossmodel = false;
		data.spritesize = 0;
		data.lightlevel = 0;
		data.skylight = 15;
		data.connectsides = false;
		data.entity=false;
		#ifndef ACCURATE_RT
		} else {
			data.lightcol = vec3(
				packedData.z % 256,
				(packedData.z >> 8) % 256,
				(packedData.z >> 16) % 256
			) / 255.0;
			data.texelcoord = ivec2(
				packedData.y % 65536,
				packedData.y >> 16
			);
			data.mat = int(packedData.x % 65536);
			uint type = packedData.z >> 24;
			data.alphatest = (type % 2 == 1);
			data.crossmodel = ((type >> 1) % 2 == 1);
			data.full = ((type >> 2) % 2 == 1);
			data.emissive = ((type >> 3) % 2 == 1);
			data.cuboid = ((type >> 4) % 2 == 1) && !data.full;
			data.trace = ((type >> 5) % 2 == 0);
			data.connectsides = ((type >> 6) % 2 == 1);
			data.entity = ((type >> 7) % 2 == 1);
			data.spritesize = (1 << ((packedData.w >> 24) % 16));
			data.lightlevel = int(packedData.x >> 16);
			data.skylight = int(packedData.w >> 28) % 16;
			if (data.cuboid) {
				data.lower = vec3(
					packedData.w % 16,
					(packedData.w >> 4) % 16,
					(packedData.w >> 8) % 16
				) / 16.0;
				data.upper = (vec3(
					(packedData.w >> 12) % 16,
					(packedData.w >> 16) % 16,
					(packedData.w >> 20) % 16
					) + 1) / 16.0;
			} else {
				data.lower = vec3(0);
				data.upper = vec3(1);
			}
			if (data.crossmodel || data.entity) {
				data.midcoord = vec3(packedData.w % 256, (packedData.w >> 8) % 256, (packedData.w >> 16) % 256) / 256.0;
			} else {
				data.midcoord = vec3(0.5);
			}
		}
	#endif
	return data;
}

vxData readVxMap(vec3 vxPos) {
	ivec3 coord = ivec3(vxPos + pointerGridSize * POINTER_VOLUME_RES / 2);
	return readVxMap(coord);
}
#endif