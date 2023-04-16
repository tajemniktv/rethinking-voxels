#include "/lib/common.glsl"

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);
uniform vec3 cameraPosition;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform sampler2D colortex8;
uniform sampler2D colortex15;

const ivec2[8] offsets = ivec2[8](
	ivec2( 1, 0),
	ivec2( 1, 1),
	ivec2( 0, 1),
	ivec2(-1, 1),
	ivec2(-1, 0),
	ivec2(-1,-1),
	ivec2( 0,-1),
	ivec2( 1,-1)
);

#include "/lib/vx/raytrace.glsl"

void main() {
	vec4 normalDepthData = texelFetch(colortex8, ivec2(gl_FragCoord.xy), 0);
	if (normalDepthData.w > 1.5) {
		normalDepthData = vec4(0);
		vec4 maxNormDD = vec4(-100);
		vec4 minNormDD = vec4(100);
		int counter = 0;
		for (int i = 0; i < 8; i++) {
			vec4 aroundData = texelFetch(colortex8, ivec2(gl_FragCoord.xy) + offsets[i], 0);
			if (aroundData.w < 1.5) {
				if (i % 2 == 0) {
					maxNormDD = max(maxNormDD, aroundData);
					minNormDD = min(minNormDD, aroundData);
				}
				normalDepthData += aroundData;
				counter++;
			}
		}
		if (maxNormDD.x < -99) {
			minNormDD = vec4(0);
			maxNormDD = vec4(0);
		}
		if (counter > 3 && length(maxNormDD - minNormDD) < 0.4) {
			normalDepthData /= counter;
		} else normalDepthData = vec4(2);
	}
	if (normalDepthData.w > 1.5) {
		vec4 clipPos = vec4(gl_FragCoord.xy / view * 2 - 1, 0.9998, 1);
		vec4 dir = gbufferModelViewInverse * (gbufferProjectionInverse * clipPos);
		dir /= dir.w;
		vec3 vxPlayerPos = fract(cameraPosition) + gbufferModelViewInverse[3].xyz;
		ray_hit_t rayHit;
		#ifdef ACCURATE_RT
			rayHit = betterRayTrace(vxPlayerPos + normalize(dir.xyz), dir.xyz, colortex15);
		#else
			rayHit = raytrace(vxPlayerPos, dir.xyz, colortex15);
		#endif
		normalDepthData.xyz = rayHit.normal;
		vec4 hitScreenPos = gbufferProjection * (gbufferModelView * vec4(rayHit.pos - vxPlayerPos, 1));
		normalDepthData.w = -hitScreenPos.z / hitScreenPos.w;
		normalDepthData = 0.5 + 0.5 * normalDepthData;
	}
	normalDepthData.xyz = 2 * normalDepthData.xyz - 1;
	/*RENDERTARGETS:8*/
	gl_FragData[0] = normalDepthData;
}