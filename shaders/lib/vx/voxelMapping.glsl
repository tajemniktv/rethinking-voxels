#ifndef MAPPING
#define MAPPING
#include "/lib/vx/SSBOs.glsl"
//// needs uniform vec3 cameraPosition, previousCameraPosition

// inverse function to getVxPixelCoords

// get voxel space position from world position
vec3 getVxPos(vec3 worldPos) {
	return worldPos + 8.0 * fract(0.125 * cameraPosition);
}

// get previous voxel space position from world position
vec3 getPreviousVxPos(vec3 worldPos) {
	return getVxPos(worldPos) + 8 * (floor(0.125 * cameraPosition) - floor(0.125 * previousCameraPosition));
}

// determine if a position is within the voxelisation range
bool isInRange(vec3 pos, float margin) {
	return all(lessThan(abs(pos) + margin, POINTER_VOLUME_RES * pointerGridSize / 2.0));
}
bool isInRange(vec3 pos) {
	return isInRange(pos, 0);
}

#endif