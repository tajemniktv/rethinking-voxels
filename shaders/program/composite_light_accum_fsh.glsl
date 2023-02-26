#include "/lib/common.glsl"

#if BL_SHADOW_MODE == 1
flat in mat4 reprojectionMatrix;
in vec2 texCoord;

uniform int frameCounter;

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;
uniform sampler2D colortex2;
uniform sampler2D colortex4;
uniform sampler2D colortex12;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform float near;
uniform float far;

float GetLinearDepth(float depth) {
	return (2.0 * near) / (far + near - depth * (far - near));
}

ivec2 offsets[4] = ivec2[4](ivec2(0, 0), ivec2(0, 1), ivec2(1, 1), ivec2(1, 0));
#endif
void main() {
	#if BL_SHADOW_MODE == 1
	ivec2 pixelCoord = ivec2(gl_FragCoord.xy);
	vec2 HRTexCoord = (pixelCoord - offsets[frameCounter % 4]) / (2.0 * view);
	vec3 col = texture2D(colortex4, HRTexCoord).rgb;
	float depth = texelFetch(depthtex1, pixelCoord, 0).r;
	vec4 prevPos = reprojectionMatrix * (vec4(texCoord, depth, 1) * 2 - 1);
	prevPos = prevPos * 0.5 / prevPos.w + 0.5;
	vec4 prevCol = texture2D(colortex12, prevPos.xy);
	float blendFactor = float(prevPos.x > 0.0 && prevPos.x < 1.0 &&
	                          prevPos.y > 0.0 && prevPos.y < 1.0);
	float prevDepth0 = GetLinearDepth(prevPos.z);
	float prevDepth1 = GetLinearDepth(texture2D(colortex12, prevPos.xy).a);
	float ddepth = abs(prevDepth0 - prevDepth1) / abs(prevDepth0);
	float offCenterLength = length(fract(view * HRTexCoord) - 0.5);
	blendFactor *= clamp(0.5 + 0.5 * offCenterLength - 3 * float(ddepth > 0.2), 0, 1);
	col = mix(col, prevCol.xyz, blendFactor);
	#else
	vec3 col = vec3(0);
	float depth = 1.0;
	#endif
	/*RENDERTARGETS:12*/
	gl_FragData[0] = vec4(col, depth);
	return;
}