#include "/lib/common.glsl"

flat in mat4 reprojectionMatrix;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform sampler2D colortex2;

layout(r32ui) uniform uimage2D colorimg9;

void main() {
	#ifdef PER_BLOCK_LIGHT
		return;
	#endif
	float prevDepth = 1 - texelFetch(colortex2, ivec2(gl_FragCoord.xy), 0).w;
	vec4 prevClipPos = vec4(gl_FragCoord.xy / view, prevDepth, 1) * 2 - 1;
	vec4 newClipPos = prevClipPos;
	if (prevDepth > 0.56) {
		newClipPos = reprojectionMatrix * prevClipPos;
		newClipPos /= newClipPos.w;
	}
	newClipPos = 0.5 * newClipPos + 0.5;
	if (prevClipPos.z > 0.99998) newClipPos.z = 0.999998;
	if (all(greaterThan(newClipPos.xyz, vec3(0))) && all(lessThan(newClipPos.xyz, vec3(0.999999)))) {
		newClipPos.xy *= view;
		vec2 diff = newClipPos.xy - gl_FragCoord.xy + 0.01;
		ivec2 writePixelCoord = ivec2(gl_FragCoord.xy + floor(diff) + 0.5);
		uint depth = uint((1<<30) * newClipPos.z);
		imageAtomicMin(colorimg9, writePixelCoord, depth);
	}
	/*DRAWBUFFERS:3*/
}