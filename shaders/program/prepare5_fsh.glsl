#include "/lib/common.glsl"

uniform float viewWidth;
uniform float viewHeight;
ivec2 view = ivec2(viewWidth + 0.1, viewHeight + 0.1);
ivec2 lowResView = view / 8;
uniform int frameCounter;
uniform vec3 cameraPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform sampler2D colortex0;
uniform sampler2D colortex3;
uniform sampler2D colortex15;

#include "/lib/vx/SSBOs.glsl"
#include "/lib/vx/raytrace.glsl"
const ivec2[4] offsets = ivec2[4](
	ivec2(0, 0),
	ivec2(1, 0),
	ivec2(1, 1),
	ivec2(0, 1)
);
void main() {
	ivec2 coords = ivec2(gl_FragCoord.xy);
	float oneMinusDepth = texelFetch(colortex0, coords, 0).a;
	ivec2 localCoords = coords * 2 + offsets[frameCounter % 4];
	vec3 lightCol = vec3(0);
	if (all(lessThan(localCoords, view))) {
		vec4 normalDepthData = texelFetch(colortex0, localCoords, 0);
		if (length(normalDepthData.xyz) > 0.5) {
			vec4 pos = vec4(localCoords, 1 - normalDepthData.w, 1);
			pos.xy = (pos.xy + 0.5) / view;
			pos.xyz = 2 * pos.xyz - 1;
			pos = gbufferModelViewInverse * (gbufferProjectionInverse * pos);
			pos.xyz = pos.xyz / pos.w + fract(cameraPosition);
			pos.xyz += 0.05 * normalDepthData.xyz;
			if (clamp(pos.xyz, -pointerGridSize / POINTER_VOLUME_RES, pointerGridSize / POINTER_VOLUME_RES) == pos.xyz) {
				ivec3 pgc = ivec3(pos.xyz / POINTER_VOLUME_RES + pointerGridSize / 2.0);
				int lightCount = PointerVolume[4][pgc.x][pgc.y][pgc.z];
				ivec2 roughCoords = localCoords / 8;
				for (int lightNum = 0; lightNum < lightCount; lightNum++) {
					ivec2 tileCoords = ivec2(lightNum % 8, lightNum / 8);
					float visible = texelFetch(colortex3, lowResView * tileCoords + roughCoords, 0).x;
					if (visible < 0.01) continue;
					light_t thisLight = lights[PointerVolume[5 + lightNum][pgc.x][pgc.y][pgc.z]];
					vec3 dir = thisLight.pos - pos.xyz;
					vec3 offset = hash33(vec3(localCoords, frameCounter)) * 1.98 - 0.99;
					offset *= thisLight.size;
					float ndotl = dot(dir, normalDepthData.xyz);
					float brightness = length(dir);
					float lightBrightness = thisLight.brightnessMat >> 16;
					brightness = ndotl * 0.0625 * lightBrightness * pow(max(0, 1 - brightness / lightBrightness), 1.5);
					if (brightness > 0.01) {
						vec3 thisLightColor = vec3(
							thisLight.packedColor % 256,
							(thisLight.packedColor >> 8) % 256,
							(thisLight.packedColor >> 16) % 256
						) / 255.0 * brightness;
						if (visible < 0.99) {
							#ifdef ACCURATE_RT
								ray_hit_t rayHit = betterRayTrace(pos.xyz, dir + offset, colortex15, false);
							#else
								ray_hit_t rayHit = raytrace(pos.xyz, dir + offset, colortex15);
							#endif
							vec3 dist = abs(thisLight.pos - rayHit.pos) / (max(thisLight.size, vec3(0.5)) + 0.01);
							if (max(max(dist.x, dist.y), dist.z) < 1.0) {
								rayHit.transColor.rgb = rayHit.transColor.a < 0.001 ? vec3(1.0) : rayHit.transColor.rgb;
								float rayBrightness = max(max(
									rayHit.transColor.r,
									rayHit.transColor.g),
									rayHit.transColor.b
								);
								rayHit.transColor.rgb /= sqrt(rayBrightness);
								rayHit.transColor.rgb *= clamp(4 - 4 * rayHit.transColor.a, 0, 1);
								thisLightColor *= rayHit.transColor.rgb;
							} else thisLightColor = vec3(0);
						}
						lightCol += thisLightColor;
					}
				}
			}
		}
	}
	/*RENDERTARGETS:0*/
	gl_FragData[0] = vec4(log(lightCol * 0.2 + 1), oneMinusDepth);
}