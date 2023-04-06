#include "/lib/common.glsl"

flat in mat4 reprojectionMatrix;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform sampler2D colortex2;
uniform sampler2D colortex4;

layout(rgba16f) uniform image2D colorimg0;

void main() {
	float prevDepth = texelFetch(colortex2, ivec2(gl_FragCoord.xy), 0).w;
	vec4 prevClipPos = vec4(gl_FragCoord.xy / view, prevDepth, 1) * 2 - 1;
	vec4 newClipPos = reprojectionMatrix * prevClipPos;
	newClipPos /= newClipPos.w;
	newClipPos = 0.5 * newClipPos + 0.5;
	if (prevClipPos.z > 0.99998) newClipPos.z = 0.999998;
	if (prevClipPos.z > 0.56 && all(greaterThan(newClipPos.xyz, vec3(0))) && all(lessThan(newClipPos.xyz, vec3(0.999999)))) {
		newClipPos.xy *= view;
		vec2 diff = newClipPos.xy - gl_FragCoord.xy + 0.01;
		ivec2 writePixelCoord = ivec2(gl_FragCoord.xy + floor(diff));
		vec2 prevSampleCoord = (gl_FragCoord.xy - fract(diff)) / view;
		vec4 writeData = vec4(texture(colortex4, prevSampleCoord).xyz, 1 - newClipPos.z);
		//if (imageLoad(colorimg0, writePixelCoord).w > newClipPos.z) {
			imageStore(colorimg0, writePixelCoord, writeData);
		//}
	}
	/*DRAWBUFFERS:3*/
}