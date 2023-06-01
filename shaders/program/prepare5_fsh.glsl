#include "/lib/common.glsl"

uniform float viewWidth;
uniform float viewHeight;
ivec2 view = ivec2(viewWidth + 0.1, viewHeight + 0.1);
ivec2 lowResView = view / 8;
uniform sampler2D colortex3;
#define BLUR_SIZE 1
void main() {
	#ifdef PER_BLOCK_LIGHT
		return;
	#endif
	ivec2 coords = ivec2(gl_FragCoord.xy);
	ivec2 tileCoords = coords / lowResView;
	float visibility = 0;
	if (all(lessThan(tileCoords, ivec2(8)))) {
		ivec2 localCoords = coords % lowResView;
		ivec2 lowerBound = max(ivec2(-BLUR_SIZE), -localCoords);
		ivec2 upperBound = min(ivec2(BLUR_SIZE + 2), lowResView - localCoords);
		for (int x = lowerBound.x; x < upperBound.x; x++) {
			for (int y = lowerBound.y; y < upperBound.y; y++) {
				visibility += texelFetch(colortex3, coords + ivec2(x, y), 0).x;
			}
		}
		visibility /= (upperBound.x - lowerBound.x) * (upperBound.y - lowerBound.y);
	}
	/*RENDERTARGETS:3*/
	gl_FragData[0] = vec4(visibility, 0, 0, 1);
}