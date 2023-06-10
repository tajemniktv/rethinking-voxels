#include "/lib/common.glsl"

flat in mat4 reprojectionMatrix;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);
uniform sampler2D colortex1;
uniform sampler2D colortex8;

void main() {
	#ifdef PER_BLOCK_LIGHT
		return;
	#endif
	vec4 pos = vec4(
		gl_FragCoord.xy / view * 2 - 1,
		1 - texelFetch(colortex8, ivec2(gl_FragCoord.xy), 0).w,
		1);
	vec4 prevPos = reprojectionMatrix * pos;
	prevPos /= prevPos.w;
	prevPos.xy = 0.5 * prevPos.xy + 0.5;
	vec4 materialData = texelFetch(colortex1, ivec2(prevPos.xy * view), 0);
	/*RENDERTARGETS:1*/
	gl_FragData[0] = materialData;
}