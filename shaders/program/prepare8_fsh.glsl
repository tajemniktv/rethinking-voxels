#include "/lib/common.glsl"

flat in mat4 reprojectionMatrix;

uniform int frameCounter;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform sampler2D colortex8;
uniform sampler2D colortex12;

uniform float near;
uniform float far;

float GetLinearDepth(float depth) {
	return (2.0 * near) / (far + near - depth * (far - near));
}

const ivec2[4] offsets = ivec2[4](
	ivec2(0, 0),
	ivec2(1, 0),
	ivec2(1, 1),
	ivec2(0, 1)
);

float getEdge(float linDepth0, vec2 coords) {
	float maxLinDepth = linDepth0;
	float minLinDepth = linDepth0;
	for (int k = 0; k < 9; k++) {
		if (k == 4) continue;
		vec2 offset = (vec2(k % 3, k / 3) - vec2(1)) / view;
		float aroundDepth = GetLinearDepth(1 - texture2D(colortex12, coords + offset).a);
		maxLinDepth = max(maxLinDepth, aroundDepth);
		minLinDepth = min(minLinDepth, aroundDepth);
	}
	maxLinDepth = min(maxLinDepth, 0.5);
	return clamp(3 * (0.5 - maxLinDepth) * (1 - abs(minLinDepth / maxLinDepth)), 0, 1);
}
void main() {
	#ifdef PER_BLOCK_LIGHT
		return;
	#endif
	ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
	vec2 HRTexCoord = (pixelCoord - offsets[frameCounter % 4]) / (2.0 * view);
	vec3 color = texture2D(colortex8, HRTexCoord).rgb;
	float depth = 1 - texelFetch(colortex8, pixelCoord, 0).w;
	vec4 prevPos = reprojectionMatrix * (vec4(gl_FragCoord.xy / view, depth, 1) * 2 - 1);
	prevPos = prevPos * 0.5 / prevPos.w + 0.5;
	vec4 prevCol = texture2D(colortex12, prevPos.xy);
	float prevDepth0 = GetLinearDepth(prevPos.z);
	float prevDepth1 = GetLinearDepth(1 - prevCol.a);
	float edge = getEdge(prevDepth1, prevPos.xy);
	float blendFactor = max(0, 1 - 100 * edge * length(cameraPosition - previousCameraPosition)) * float(prevPos.x > 0.0 && prevPos.x < 1.0 &&
	                                 prevPos.y > 0.0 && prevPos.y < 1.0);
	float ddepth = abs(prevDepth0 - prevDepth1) / abs(prevDepth0);
	float offCenterLength = length(fract(view * HRTexCoord) - 0.5);
	blendFactor *= 0.5 + 0.5 * offCenterLength - 3 * float(ddepth > 0.1);
	blendFactor = clamp(blendFactor, 0, 1);
	color = mix(color, prevCol.xyz, blendFactor);
	/*RENDERTARGETS:1,12*/
	gl_FragData[0] = vec4(1);
	gl_FragData[1] = vec4(color, 1 - depth);
}