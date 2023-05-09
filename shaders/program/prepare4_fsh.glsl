#include "/lib/common.glsl"

uniform int frameCounter;

uniform float viewWidth;
uniform float viewHeight;
ivec2 view = ivec2(viewWidth + 0.1, viewHeight + 0.1);
ivec2 lowResView = view / 8;
uniform vec3 fogColor;
uniform vec3 cameraPosition;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform sampler2D colortex1;
uniform sampler2D colortex8;
uniform sampler2D colortex15;

#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/raytrace.glsl"

void main() {
	ivec2 coords = ivec2(gl_FragCoord.xy);
	ivec2 tileCoords = coords / lowResView;
	float visibility = 0;
	vec4 debug = vec4(0);
	if (all(lessThan(tileCoords, ivec2(8)))) {
		int lightNum = tileCoords.x + 8 * tileCoords.y;
		ivec2 localCoords = coords % lowResView * 8;
		if (all(lessThan(localCoords, view))) {
			vec4 normalDepthData = texelFetch(colortex8, localCoords, 0);
			vec4 materialData = texelFetch(colortex1, localCoords, 0);
			if (length(normalDepthData.xyz) > 0.5) {
				vec4 pos = vec4(localCoords, 1 - normalDepthData.w, 1);
				pos.xy = (pos.xy + 0.5) / view;
				pos.xyz = 2 * pos.xyz - 1;
				pos = gbufferModelViewInverse * (gbufferProjectionInverse * pos);
				pos.xyz = pos.xyz / pos.w + fract(cameraPosition);
				debug.xyz = (pos.xyz + floor(cameraPosition) - vec3(50, 65, 0)) * 0.1;
				float posLen = length(pos);
				pos.xyz += max(0.05, 0.01 * posLen) * normalDepthData.xyz;
				if (clamp(pos.xyz, -pointerGridSize * POINTER_VOLUME_RES / 2.0, pointerGridSize * POINTER_VOLUME_RES / 2.0) == pos.xyz) {
					ivec3 pgc = ivec3(pos.xyz / POINTER_VOLUME_RES + pointerGridSize / 2.0) / 4;
					int lightCount = readVolumePointer(pgc, 4);
					int lightStripLoc = readVolumePointer(pgc, 5) + 1;
					if (lightCount > lightNum) {
						light_t thisLight = lights[readLightPointer(lightStripLoc + lightNum)];
						vec3 dir = thisLight.pos - pos.xyz;
						vec3 offset = hash33(vec3(localCoords, frameCounter)) * 1.98 - 0.99;
						vec3 sssOffset = materialData.a < 0.75 ? max(posLen * 0.03, 0.3) * normalize(dir) : vec3(0);
						offset *= thisLight.size;
						if (dot(dir, normalDepthData.xyz) > 0 || materialData.a < 0.75) {
							#ifdef ACCURATE_RT
								ray_hit_t rayHit = betterRayTrace(pos.xyz + sssOffset, dir + offset - sssOffset, colortex15, false);
							#else
								ray_hit_t rayHit = raytrace(pos.xyz + sssOffset, dir + offset - sssOffset, colortex15);
							#endif
							if (rayHit.rayColor.a < 0.1) {
								visibility = 1.0;
							} else if (rayHit.rayColor.a < 0.9) {
								visibility = 0.5;
							} else {
								vec3 dist = abs(rayHit.pos - thisLight.pos) / (max(thisLight.size, vec3(0.5)) + 0.01);
								if (max(max(dist.x, dist.y), dist.z) > 1.0) visibility = 0.0;
								else if (rayHit.transColor.a > 0.1) visibility = 0.5;
								else visibility = 1.0;
							}
						}
					}
				}
			}
		}
	}
	/*RENDERTARGETS:3*/
	gl_FragData[0] = vec4(visibility, 0, 0, 1);
}