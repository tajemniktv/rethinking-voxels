#ifndef RAYTRACE
#define RAYTRACE
#include "/lib/vx/voxelMapping.glsl"
#include "/lib/vx/voxelReading.glsl"
#if CAVE_SUNLIGHT_FIX > 0
#ifndef COLORTEX10
#define COLORTEX10
uniform sampler2D colortex10;
#endif
#endif
#ifdef DISTANCE_FIELD
#ifndef COLORTEX11
#define COLORTEX11
uniform sampler2D colortex11;
#endif
#endif

const mat3 eye = mat3(
	1, 0, 0,
	0, 1, 0,
	0, 0, 1
);
// cuboid intersection algorithm
float aabbIntersect(vxData data, vec3 pos, vec3 dir, inout int n) {
	// offset to work around floating point errors
	vec3 offset = 0.001 * eye[n] * sign(dir[n]);
	// for connected blocks like walls, fences etc, figure out connection sides
	bool renderMainCuboid = true;
	bvec2 renderConnectCuboids = bvec2(false);
	vec3[4] connectCuboids = vec3[4](
		vec3(min(data.lower + 0.0625, 0.4375)),
		vec3(max(data.upper - 0.0625, 0.5625)),
		vec3(min(data.lower + 0.0625, 0.4375)),
		vec3(max(data.upper - 0.0625, 0.5625)));
	if (data.connectsides) {
		for (int k = 0; k < 4; k++) {
			connectCuboids[k].y = (k % 2 == 0) ? (abs(data.lower.x - 0.375) < 0.01 ? 0.375 : 0.0) : (abs(data.lower.x - 0.25) < 0.01 ? 0.875 : (abs(data.lower.x - 0.375) < 0.01 ? 0.9375 : 1.0));
			vec3 blockOffset = vec3(k % 2 * 2 - 1) * vec3(1 - (k >> 1), 0, k >> 1);
			vec3 thisOffsetPos = pos + offset + blockOffset;
			if (isInRange(thisOffsetPos)) {
				vxData offsetData = readVxMap(thisOffsetPos);
				if ((offsetData.connectsides && !(abs(offsetData.lower.x - 0.375) < 0.01 ^^ abs(data.lower.x - 0.375) < 0.01)) || (offsetData.full && !offsetData.alphatest)) {
					connectCuboids[k][2 * (k >> 1)] = k % 2;
					renderConnectCuboids[k >> 1] = true;
				}
			}
		}
		if (abs(data.lower.x - 0.25) < 0.01 && ((renderConnectCuboids == bvec2(true, false) && connectCuboids[0].x < 0.01 && connectCuboids[1].x > 0.99) || (renderConnectCuboids == bvec2(false, true) && connectCuboids[2].z < 0.01 && connectCuboids[3].z > 0.99))) renderMainCuboid = false;
	}
	// don't need to know global position, only relative to current block
	pos = fract(pos + offset) - offset;
	float w = 10000;
	for (int k = 0; k < 2; k++) {
		if (renderConnectCuboids[k]) {
			for (int i = 0; i < 3; i++) {
				if (dir[i] == 0) continue;
				for (int l = 0; l < 2; l++) {
					float w0 = (connectCuboids[2 * k + l][i] - pos[i]) / dir[i];
					// ray-plane intersection position needs to be closer than the previous best one and further than approximately 0
					bool valid = (w0 > -0.00005 / length(dir) && w0 < w);
					if (!valid) break;
					vec3 newPos = pos + w0 * dir;
					for (int j = 1; j < 3; j++) {
						int ij = (i + j) % 3;
						// intersection position also needs to be within other bounds
						if (newPos[ij] < connectCuboids[2 * k][ij] || newPos[ij] > connectCuboids[2 * k + 1][ij]) {
							valid = false;
							break;
						}
					}
					// update normal and ray position
					if (valid) {
						w = w0;
						n = i;
					}					
				}
			}
		}
	}
	if (renderMainCuboid) {
	vec3[2] bounds = vec3[2](data.lower, data.upper);
		for (int i = 0; i < 3; i++) {
			if (dir[i] == 0) continue;
			float relevantBound = bounds[dir[i] < 0 ? 1 : 0][i];
			float w0 = (relevantBound - pos[i]) / dir[i];
			if (w0 < -0.00005 / length(dir)) {
				relevantBound = bounds[dir[i] < 0 ? 0 : 1][i];
				w0 = (relevantBound - pos[i]) / dir[i];
			}
			vec3 newPos = pos + w0 * dir;
			// ray-plane intersection position needs to be closer than the previous best one and further than approximately 0
			bool valid = (w0 > -0.00005 / length(dir) && w0 < w);
			for (int j = 1; j < 3; j++) {
				int ij = (i + j) % 3;
				// intersection position also needs to be within other bounds
				if (newPos[ij] < bounds[0][ij] || newPos[ij] > bounds[1][ij]) {
					valid = false;
					break;
				}
			}
			// update normal and ray position
			if (valid) {
				w = w0;
				n = i;
			}
		}
	}
	return w;
}
// returns color data of the block at pos, when hit by ray in direction dir
vec4 handledata(vxData data, sampler2D atlas, inout vec3 pos, vec3 dir, int n) {
	if (!data.crossmodel) {
		if (data.cuboid) {
			float w = aabbIntersect(data, pos, dir, n);
			if (w > 9999) return vec4(0);
			pos += w * dir;
		}
		vec2 spritecoord = vec2(n != 0 ? fract(pos.x) : fract(pos.z), n != 1 ? fract(-pos.y) : fract(pos.z)) * 2 - 1;
		ivec2 texcoord = ivec2(data.texcoord * atlasSize + (data.spritesize - 0.5) * spritecoord);
		vec4 color = texelFetch(atlas, texcoord, 0);
		if (!data.alphatest) color.a = 1;
		else if (color.a > 0.1 && color.a < 0.9) color.a = min(pow(color.a, TRANSLUCENT_LIGHT_TINT), 0.8);
		// multiply by vertex color for foliage, water etc
		color.rgb *= data.emissive ? vec3(1) : data.lightcol;
		return color;
	}
	// get around floating point errors using an offset
	vec3 offset = 0.001 * eye[n] * sign(dir[n]);
	vec3 blockInnerPos0 = fract(pos + offset) - offset;
	vec3 blockInnerPos = blockInnerPos0 - vec3(data.midcoord.x, 0, data.midcoord.z);;
	// ray-plane intersections
	float w0 = (-blockInnerPos.x - blockInnerPos.z) / (dir.x + dir.z);
	float w1 = (blockInnerPos.x - blockInnerPos.z) / (dir.z - dir.x);
	vec3 p0 = blockInnerPos + w0 * dir + vec3(0.5, 0, 0.5);
	vec3 p1 = blockInnerPos + w1 * dir + vec3(0.5, 0, 0.5);
	bool valid0 = (max(max(abs(p0.x - 0.5), 0.8 * abs(p0.y - 0.5)), abs(p0.z - 0.5)) < 0.4) && w0 > -0.0001;
	bool valid1 = (max(max(abs(p1.x - 0.5), 0.8 * abs(p1.y - 0.5)), abs(p1.z - 0.5)) < 0.4) && w1 > -0.0001;
	vec4 color0 = valid0 ? texelFetch(atlas, ivec2(data.texcoord * atlasSize + (data.spritesize - 0.5) * (1 - p0.xy * 2)), 0) : vec4(0);
	vec4 color1 = valid1 ? texelFetch(atlas, ivec2(data.texcoord * atlasSize + (data.spritesize - 0.5) * (1 - p1.xy * 2)), 0) : vec4(0);
	color0.xyz *= data.emissive ? vec3(1) : data.lightcol;
	color1.xyz *= data.emissive ? vec3(1) : data.lightcol;
	pos += (valid0 ? w0 : (valid1 ? w1 : 0)) * dir;
	// the more distant intersection position only contributes by the amount of light coming through the closer one
	return (w0 < w1) ? (vec4(color0.xyz * color0.a, color0.a) + (1 - color0.a) * vec4(color1.xyz * color1.a, color1.a)) : (vec4(color1.xyz * color1.a, color1.a) + (1 - color1.a) * vec4(color0.xyz * color0.a, color0.a));
}
// voxel ray tracer
vec4 raytrace(bool lowDetail, inout vec3 pos0, bool doScattering, vec3 dir, inout vec3 translucentHit, sampler2D atlas, bool translucentData) {
	ivec3 dcamPos = ivec3(1.001 * (floor(cameraPosition) - floor(previousCameraPosition)));
	vec3 progress;
	for (int i = 0; i < 3; i++) {
		//set starting position in each direction
		progress[i] = -(dir[i] < 0 ? fract(pos0[i]) : fract(pos0[i]) - 1) / dir[i];
	}
	int i = 0;
	// get closest starting position
	float w = progress[0];
	for (int i0 = 1; i0 < 3; i0++) {
		if (progress[i0] < w) {
			i = i0;
			w = progress[i];
		}
	}
	// step size in each direction (to keep to the voxel grid)
	vec3 stp = abs(1 / dir);
	float dirlen = length(dir);
	float invDirLenScaled = 0.001 / dirlen;
	vec3 dirsgn = sign(dir);
	vec3[3] eyeOffsets;
	for (int k = 0; k < 3; k++) {
		eyeOffsets[k] = 0.0001 * eye[k] * dirsgn[k];
	}
	vec3 pos = pos0 + invDirLenScaled * dir;
	vec3 scatterPos = pos0;
	vec4 rayColor = vec4(0);
	vec4 oldRayColor = vec4(0);
	const float scatteringMaxAlpha = 0.1;
	// check if stuff already needs to be done at starting position
	vxData voxeldata = readVxMap(getVxPixelCoords(pos));
	bool isScattering = false;
	if (lowDetail && voxeldata.full && !voxeldata.alphatest) return vec4(0, 0, 0, translucentData ? 0 : 1);
	if (isInRange(pos) && voxeldata.trace && !lowDetail) {
		rayColor = handledata(voxeldata, atlas, pos, dir, i);
		if (dot(pos - pos0, dir / dirlen) <= 0.01) rayColor.a = 0;
		if (doScattering && rayColor.a > 0.1) isScattering = (voxeldata.mat == 10004 || voxeldata.mat == 10008 || voxeldata.mat == 10016);
		if (doScattering && isScattering) {
			scatterPos = pos;
			rayColor.a = min(scatteringMaxAlpha, rayColor.a);
		}
		rayColor.rgb *= rayColor.a;
	}
	if (rayColor.a > 0.01 && rayColor.a < 0.9) translucentHit = pos;
	int k = 0; // k is a safety iterator
	int mat = rayColor.a > 0.1 ? voxeldata.mat : 0; // for inner face culling
	vec3 oldPos = pos;
	bool oldFull = voxeldata.full;
	bool wasInRange = false;
	// main loop
	while (w < 1 && k < 2000 && rayColor.a < 0.999) {
		oldRayColor = rayColor;
		pos = pos0 + (min(w, 1.0)) * dir + eyeOffsets[i];
		#ifdef DISTANCE_FIELD
		ivec4 dfdata;
		#endif
		// read voxel data at new position and update ray colour accordingly
		if (isInRange(pos)) {
			wasInRange = true;
			ivec2 vxCoords = getVxPixelCoords(pos);
			voxeldata = readVxMap(vxCoords);
			#ifdef DISTANCE_FIELD
			#ifdef FF_IS_UPDATED
			ivec2 oldCoords = vxCoords;
			#else
			ivec2 oldCoords = getVxPixelCoords(pos + dcamPos);
			#endif
			dfdata = ivec4(texelFetch(colortex11, oldCoords, 0) * 65525 + 0.5);
			#endif
			pos -= eyeOffsets[i];
			if (lowDetail) {
				if (voxeldata.trace && voxeldata.full && !voxeldata.alphatest) {
					pos0 = pos + eyeOffsets[i];
					return vec4(0, 0, 0, translucentData ? 0 : 1);
				}
			} else {
				bool newScattering = false;
				if (voxeldata.trace) {
					vec4 newColor = handledata(voxeldata, atlas, pos, dir, i);
					if (dot(pos - pos0, dir) < 0.0) newColor.a = 0;
					bool samemat = voxeldata.mat == mat;
					mat = (newColor.a > 0.1) ? voxeldata.mat : 0;
					if (doScattering) newScattering = (mat == 10004 || mat == 10008 || mat == 10016);
					if (newScattering) newColor.a = min(newColor.a, scatteringMaxAlpha);
					if (samemat) newColor.a = clamp(10.0 * newColor.a - 9.0, 0.0, 1.0);
					rayColor.rgb += (1 - rayColor.a) * newColor.a * newColor.rgb;
					rayColor.a += (1 - rayColor.a) * newColor.a;
					if (oldRayColor.a < 0.01 && rayColor.a > 0.01 && rayColor.a < 0.9) translucentHit = pos;
				}
				if (doScattering) {
					if (isScattering) {
						scatterPos = pos;
					}
					oldFull = voxeldata.full;
					oldPos = pos;
					isScattering = newScattering;
				}
			}
			#if CAVE_SUNLIGHT_FIX > 0
			if (!isInRange(pos, 2)) {
				int height0 = int(texelFetch(colortex10, ivec2(pos.xz + floor(cameraPosition.xz) - floor(previousCameraPosition.xz) + vxRange / 2), 0).w * 65535 + 0.5) % 256 - VXHEIGHT * VXHEIGHT / 2;
				if (pos.y + floor(cameraPosition.y) - floor(previousCameraPosition.y) < height0) {
					rayColor.a = 1;
				}
			}
			#endif
			pos += eyeOffsets[i];
		}
		else {
			#ifdef DISTANCE_FIELD
			dfdata.x = int(max(max(abs(pos.x), abs(pos.z)) - vxRange / 2, abs(pos.y) - VXHEIGHT * VXHEIGHT / 2) + 0.5);
			#endif
			if (wasInRange) break;
		}
		// update position
		#ifdef DISTANCE_FIELD
		if (dfdata.x % 256 == 0) dfdata.x++;
		for (int j = 0; j < dfdata.x % 256; j++) {
		#endif
			progress[i] += stp[i];
			w = progress[0];
			i = 0;
			for (int i0 = 1; i0 < 3; i0++) {
				if (progress[i0] < w) {
					i = i0;
					w = progress[i];
				}
			}
		#ifdef DISTANCE_FIELD
		}
		#endif
		k++;
	}
	float oldAlpha = rayColor.a;
	rayColor.a = 1 - exp(-4*length(scatterPos - pos0)) * (1 - rayColor.a);
	rayColor.rgb += rayColor.a - oldAlpha; 
	pos0 = pos;
	if (k == 2000) {
		oldRayColor = vec4(1, 0, 0, 1);
		rayColor = vec4(1, 0, 0, 1);
	}
	return translucentData ? oldRayColor : rayColor;
}

vec4 raytrace(inout vec3 pos0, bool doScattering, vec3 dir, sampler2D atlas, bool translucentData) {
	vec3 translucentHit = vec3(0);
	return raytrace(false, pos0, doScattering, dir, translucentHit, atlas, translucentData);
}

vec4 raytrace(bool lowDetail, inout vec3 pos0, vec3 dir, inout vec3 translucentHit, sampler2D atlas, bool translucentData) {
	return raytrace(lowDetail, pos0, false, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(inout vec3 pos0, bool doScattering, vec3 dir, sampler2D atlas) {
	vec3 translucentHit = vec3(0);
	return raytrace(false, pos0, doScattering, dir, translucentHit, atlas, false);
}
vec4 raytrace(inout vec3 pos0, vec3 dir, inout vec3 translucentHit, sampler2D atlas, bool translucentData) {
	return raytrace(false, pos0, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(bool lowDetail, inout vec3 pos0, vec3 dir, sampler2D atlas) {
	vec3 translucentHit = vec3(0);
	return raytrace(lowDetail, pos0, dir, translucentHit, atlas, false);
}
vec4 raytrace(bool lowDetail, inout vec3 pos0, vec3 dir, sampler2D atlas, bool translucentData) {
	vec3 translucentHit = vec3(0);
	return raytrace(lowDetail, pos0, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(inout vec3 pos0, vec3 dir, sampler2D atlas, bool translucentData) {
	vec3 translucentHit = vec3(0);
	return raytrace(pos0, dir, translucentHit, atlas, translucentData);
}
vec4 raytrace(inout vec3 pos0, vec3 dir, sampler2D atlas) {
	return raytrace(pos0, dir, atlas, false);
}



struct ray_hit_t {
	vec4 rayColor;
	vec4 transColor;
	int triId;
	int transTriId;
	vec3 pos;
	vec3 transPos;
};

vec4 raytrace(vec3 pos0, vec3 dir, sampler2D atlas, inout ray_hit_t rayHit) {
	vec3 transPos = vec3(-10000);
	vec4 rayColor = raytrace(pos0, dir, transPos, atlas, true);
	rayHit.rayColor = vec4(0);
	rayHit.transColor = rayColor;
	rayHit.triId = -1;
	rayHit.transTriId = -1;
	rayHit.pos = pos0;
	rayHit.transPos = transPos;
	return rayColor;
}

bool isInBounds(vec3 v, vec3 lower, vec3 upper) {
	if (v == clamp(v, lower, upper)) return true;
	return false;
}

#include "/lib/vx/SSBOs.glsl"

float getNiceness(mat3 A) {
	return abs(A[0][0] * A[1][1] * A[2][2]);
}

vec3 linSolve(mat3 A, vec3 b) {
	A = transpose(A);
	for (int i = 0; i < 2; i++) {
		for (int j = i + 1; j < 3; j++) {
			if (abs(A[j][i]) > abs(A[i][i])) {
				vec3 tmp = A[i];
				A[i] = A[j];
				A[j] = tmp;
				float tmp2 = b[i];
				b[i] = b[j];
				b[j] = tmp2;
			}
		}
		for (int j = i + 1; j < 3; j++) {
			float t = A[j][i] / A[i][i];
			A[j] -= t * A[i];
			b[j] -= t * b[i];
		}
	}
	for (int j = 1; j >= 0; j--) {
		for (int i = j + 1; i < 3; i++) {
			float t = A[j][i] / A[i][i];
			A[j][i] = 0;
			b[j] -= t * b[i];
		}
	}
	return vec3(
		b[0] / A[0][0],
		b[1] / A[1][1],
		b[2] / A[2][2]
	);
}

vec3 rayTriangleIntersect(vec3 pos0, vec3 dir, tri_t triangle) {
	mat3 solveMat = mat3(triangle.pos[1] - triangle.pos[0], triangle.pos[2] - triangle.pos[0], -dir);
	vec3 solveVec = pos0 - triangle.pos[0];
	vec3 solution = linSolve(solveMat, solveVec);
	if (min(solution.x, solution.y) < 0 || (solution.x + solution.y) > 1) return vec3(-1);
	return solution;
}

vec2 boundsIntersect(vec3 pos0, vec3 dir, vec3 lower0, vec3 upper0) {
	vec3 dirsgn = vec3(greaterThan(dir, vec3(0))) * 2.0 - 1.0;
	dir = abs(dir);
	dir += 0.0001 * (1 - sign(dir));
	lower0 -= pos0;
	upper0 -= pos0;
	lower0 *= dirsgn;
	upper0 *= dirsgn;
	vec3 lower = min(lower0, upper0);
	vec3 upper = max(lower0, upper0);
	float w0 = -100000.0;
	float w1 =  100000.0;
	for (int i = 0; i < 3; i++) {
		w0 = max(w0, lower[i] / dir[i]);
		w1 = min(w1, upper[i] / dir[i]);
	}
	if (w0 <= w1) return vec2(w0, w1);
	return vec2(1, -1);
}

vec2 boundsIntersect(vec3 pos0, vec3 dir, tri_t triangle) {
	vec3 lower0 = min(min(triangle.pos[0], triangle.pos[1]), triangle.pos[2]);
	vec3 upper0 = max(max(triangle.pos[0], triangle.pos[1]), triangle.pos[2]);
	return boundsIntersect(pos0, dir, lower0, upper0);
}

ray_hit_t betterRayTrace(vec3 pos0, vec3 dir, sampler2D atlas, bool backFaceCulling) {
	pos0 *= 1.0 / POINTER_VOLUME_RES;
	dir *= 1.0 / POINTER_VOLUME_RES;

	ray_hit_t returnVal;

	returnVal.transColor = vec4(0);
	returnVal.transPos = vec3(-10000);
	returnVal.transTriId = -1;

	vec3 progress;
	for (int i = 0; i < 3; i++) {
		//set starting position in each direction
		progress[i] = -(dir[i] < 0 ? fract(pos0[i]) : fract(pos0[i]) - 1) / dir[i];
	}
	// step size in each direction (to keep to the voxel grid)
	vec3 stp = abs(1 / dir);
	float dirlen = length(dir);
	float invDirLenScaled = 0.001 / dirlen;
	int i = 0;
	float istp = stp[0];
	for (int j = 1; j < 3; j++) {
		if (stp[j] < istp) {
			istp = stp[j];
			i = j;
		}
	}
	float w = invDirLenScaled;
	progress[i] -= stp[i];
	vec3 dirsgn = sign(dir);
	vec3[3] eyeOffsets;
	for (int k = 0; k < 3; k++) {
		eyeOffsets[k] = 0.0001 * eye[k] * dirsgn[k];
	}
	int k = 0; // k is a safety iterator
	vec3 oldPos = pos0;
	bool oldFull = false;
	bool wasInRange = false;
	vec3 pos;
	vec4 rayColor = vec4(0);
	int hitID = -1;
	float hitW = 1;
	float transHitW = 1;
	for (;w < 1 && k < 200 && rayColor.a < 0.999; k++) {
		pos = pos0 + w * dir + eyeOffsets[i];
		if (isInBounds(pos, -pointerGridSize / 2, pointerGridSize / 2)) {
			wasInRange = true;
			ivec3 coords = ivec3(pos + pointerGridSize / 2);
			int triLocHere = PointerVolume[1][coords.x][coords.y][coords.z];
			int triCountHere = PointerVolume[0][coords.x][coords.y][coords.z];
			while (triCountHere > 0) {
				int packedBounds = PointerVolume[2][coords.x][coords.y][coords.z];
				vec3 lowerBound = vec3(
					packedBounds % 32,
					(packedBounds >> 5) % 32,
					(packedBounds >> 10) % 32
				) / 31.0;
				vec3 upperBound = vec3(
					(packedBounds >> 15) % 32,
					(packedBounds >> 20) % 32,
					(packedBounds >> 25) % 32
				) / 31.0;
				vec3 lowerBound0 = lowerBound;
				vec3 upperBound0 = upperBound;
				lowerBound += coords - pointerGridSize / 2;
				upperBound += coords - pointerGridSize / 2;
				vec2 vxBoundIsct = boundsIntersect(pos0, dir, lowerBound, upperBound);
				if (vxBoundIsct.y <= 0 || vxBoundIsct.x > hitW) break;
				for (int j = 1; j <= triCountHere; j++) {
					int thisTriId = bvhLeaves[triLocHere + j];
					if (thisTriId == 0) rayColor.r += 0.01;
					tri_t thisTri = tris[thisTriId];
					if (backFaceCulling) {
						vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
						if (dot(cnormal, dir) >= 0) continue;
					}
					vec2 boundWs = boundsIntersect(POINTER_VOLUME_RES * pos0, POINTER_VOLUME_RES * dir, thisTri);
					if (boundWs.y <= 0 || boundWs.x > hitW) continue;
					vec3 hitPos = rayTriangleIntersect(POINTER_VOLUME_RES * pos0, POINTER_VOLUME_RES * dir, thisTri);
					if (hitPos.z <= 0 || hitPos.z > hitW) continue;
					ivec2 coord0 = ivec2(thisTri.texCoord[0] % 65536, thisTri.texCoord[0] / 65536);
					vec2 coord = coord0;
					vec2 offsetDir = vec2(0);
					for (int i = 0; i < 2; i++) {
						ivec2 coord1 = ivec2(thisTri.texCoord[i+1] % 65536, thisTri.texCoord[i+1] / 65536);
						vec2 dcoord = coord1 - coord0;
						dcoord += sign(dcoord);
						coord += vec2(hitPos[i] * dcoord);
						offsetDir += sign(dcoord) * (1 - abs(offsetDir));
					}
					coord -= 0.5 * offsetDir;
					vec4 newColor = ((thisTri.matBools >> 16) % 2 == 0) ? texelFetch(atlas, ivec2(coord + 0.5), 0) : vec4(1);
					if (newColor.a < 0.1) continue;
					vec4 vertexCol0 = vec4(
							thisTri.vertexCol[0] % 256,
						(thisTri.vertexCol[0] >>  8) % 256,
						(thisTri.vertexCol[0] >> 16) % 256,
						(thisTri.vertexCol[0] >> 24) % 256
					);
					vec4 vertexCol = vertexCol0;
					for (int i = 0; i < 2; i++) {
						vec4 vertexCol1 = vec4(
								thisTri.vertexCol[i+1] % 256,
							(thisTri.vertexCol[i+1] >>  8) % 256,
							(thisTri.vertexCol[i+1] >> 16) % 256,
							(thisTri.vertexCol[i+1] >> 24) % 256
						);
						vertexCol += hitPos[i] * (vertexCol1 - vertexCol0);
					}
					newColor *= vertexCol / 255.0;
					if (transHitW < hitPos.z) rayColor += (1 - rayColor.a) * newColor * vec2(1, newColor.a).yyyx;
					else           rayColor = newColor +  (1 - newColor.a) * rayColor * vec2(1, rayColor.a).yyyx;
					if (newColor.a > 0.9) {
						if (hitPos.z < hitW) {
							hitW = hitPos.z;
							returnVal.pos = pos0 + hitPos.z * dir;
							returnVal.triId = thisTriId;
						}
					} else if (hitPos.z < transHitW) {
						transHitW = hitPos.z;
						returnVal.transPos = pos0 + hitPos.z * dir;
						returnVal.transTriId = thisTriId;
						returnVal.transColor = newColor;
					}
				}
				break;
			}
		} else if (wasInRange) break;
		progress[i] += stp[i];
		w = progress[0];
		i = 0;
		for (int i0 = 1; i0 < 3; i0++) {
			if (progress[i0] < w) {
				i = i0;
				w = progress[i];
			}
		}
	}
	if (rayColor.a < 0.999) {
		returnVal.pos = pos0 + dir;
	}
	returnVal.rayColor = rayColor;
	returnVal.pos *= POINTER_VOLUME_RES;
	returnVal.transPos *= POINTER_VOLUME_RES;
	return returnVal;
}

ray_hit_t betterRayTrace(vec3 pos0, vec3 dir, sampler2D atlas) {
	return betterRayTrace(pos0, dir, atlas, true);
}

#endif