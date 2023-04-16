#include "/lib/common.glsl"

flat in mat4 reprojectionMatrix;

uniform int frameCounter;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform sampler2D colortex8;
uniform sampler2D colortex12;

uniform float near;
uniform float far;

uniform vec3 fogColor;

float GetLinearDepth(float depth) {
	return (2.0 * near) / (far + near - depth * (far - near));
}

const ivec2[4] offsets = ivec2[4](
	ivec2(0, 0),
	ivec2(1, 0),
	ivec2(1, 1),
	ivec2(0, 1)
);
void main() {
	ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
	vec2 HRTexCoord = (pixelCoord - offsets[frameCounter % 4]) / (2.0 * view);
	vec3 color = texture2D(colortex8, HRTexCoord).rgb;
	float depth = 1 - texelFetch(colortex8, pixelCoord, 0).w;
	vec4 prevPos = reprojectionMatrix * (vec4(gl_FragCoord.xy / view, depth, 1) * 2 - 1);
	prevPos = prevPos * 0.5 / prevPos.w + 0.5;
	vec4 prevCol = texture2D(colortex12, prevPos.xy);
	float blendFactor = float(prevPos.x > 0.0 && prevPos.x < 1.0 &&
	                          prevPos.y > 0.0 && prevPos.y < 1.0);
	float prevDepth0 = GetLinearDepth(prevPos.z);
	float prevDepth1 = GetLinearDepth(texture2D(colortex12, prevPos.xy).a);
	float ddepth = abs(prevDepth0 - prevDepth1) / abs(prevDepth0);
	float offCenterLength = length(fract(view * HRTexCoord) - 0.5);
	blendFactor *= clamp(0.5 + 0.5 * offCenterLength - 3 * float(ddepth > 0.2), 0, 1);
	color = mix(color, prevCol.xyz, blendFactor);
	/*RENDERTARGETS:1,12*/
	gl_FragData[0] = vec4(1);
	gl_FragData[1] = vec4(color, depth);
}