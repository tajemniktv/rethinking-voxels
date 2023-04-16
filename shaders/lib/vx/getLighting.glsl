#ifndef LIGHTING
	#define LIGHTING
	#ifndef COLORTEX12
		#define COLORTEX12
		uniform sampler2D colortex12;
	#endif

	#include "/lib/vx/voxelReading.glsl"
	#include "/lib/vx/voxelMapping.glsl"
	#include "/lib/vx/raytrace.glsl"
	#include "/lib/vx/SSBOs.glsl"
	#define DEBUG_OCCLUDERS
	vec3 newGetBlockLight(vec3 pos, vec3 normal, int mat) {
		vec3 lightCol = vec3(0);
		vec3 volumePos = 1.0 / POINTER_VOLUME_RES * pos + pointerGridSize * 0.5;
		if (isInBounds(volumePos, vec3(0), pointerGridSize - 0.01)) {
			#ifdef DEBUG_OCCLUDERS
				if (length(fract(volumePos + 0.5) - 0.5) < 0.1) return vec3(1, 0, 0);
			#endif
			ivec3 volumeCoords = ivec3(volumePos);
			int localLightCount = PointerVolume[4][volumeCoords.x][volumeCoords.y][volumeCoords.z];
			for (int i = 0; i < localLightCount; i++) {
				int thisLightId = PointerVolume[5 + i][volumeCoords.x][volumeCoords.y][volumeCoords.z];
				light_t thisLight = lights[thisLightId];
				float ndotl = max(0, 0.99 * dot(normalize(thisLight.pos - pos), normal) + 0.01);
				float brightness = length((thisLight.pos - pos));
				float lightBrightness = thisLight.brightnessMat >> 16;
				brightness = ndotl * 0.0625 * lightBrightness * pow(max(0, 1 - brightness / lightBrightness), 1.5);
				if (brightness > 0.01) {
					#ifdef CONST_RT_NOISE
						vec3 offset = vec3(0.243567, 0.823, 0.9241) * 2.0 - 1.0;
					#else
						vec3 offset = hash33(vec3(gl_FragCoord.xy, frameCounter)) * 1.98 - 0.99;
					#endif
					offset *= thisLight.size;
					ray_hit_t rayHit;
					#ifdef ACCURATE_RT
						rayHit = betterRayTrace(pos, thisLight.pos - pos + offset, ATLASTEX);
					#else
						raytrace(pos, thisLight.pos - pos + offset, ATLASTEX, rayHit);
						vxData thisLightData = readVxMap(thisLight.pos);
						thisLight.size = thisLightData.cuboid ? 0.5 * (thisLightData.upper - thisLightData.lower) : vec3(0.5);
					#endif
					vec3 dist0 = abs(rayHit.pos - thisLight.pos) / (thisLight.size + 0.01);
					float dist = max(max(dist0.x, dist0.y), dist0.z);
					if (dist < 1.0) {
						vec3 thisLightCol = 1.0 / 255.0 * vec3(thisLight.packedColor % 256, (thisLight.packedColor >> 8) % 256, (thisLight.packedColor >> 16) % 256);
						lightCol += brightness * thisLightCol * mix(vec3(1), rayHit.transColor.rgb, sqrt(rayHit.transColor.a));
					}
				}
			}
		}
		return lightCol;
	}
	#ifndef NEW_BLOCKLIGHT_ONLY
		vec3 getGI(vec3 vxPos, vec3 normal, int mat, bool doScattering) {
			vxPos += 0.5 * normal;
			vec3 lightCol = vec3(0);
			vec3 vxPosOld = vxPos + floor(cameraPosition) - floor(previousCameraPosition) - 0.5;
			vec3 floorPos = floor(vxPosOld);
			vec3 fractPos = fract(vxPosOld);
			#ifdef GI
				vec3 lightCol1 = vec3(0);
				#define NORMAL_OFFSET 0.1
				vec3 vxPosOld1 = vxPosOld + NORMAL_OFFSET * normal;
				vec3 fractPos1 = fractPos + NORMAL_OFFSET * normal;
			#endif
			float totalInt = 0.001;
			float totalInt1 = 0.001;
			for (int k = 0; k < 8; k++) {
				vec3 offset = vec3(k%2, (k>>1)%2, (k>>2)%2);
				vec3 cornerPos = floorPos + offset;
				if (!isInRange(cornerPos)) continue;
				ivec2 cornerVxCoordsFF = getVxPixelCoords(cornerPos);
				vec4 cornerLightData = vec4(0);//texelFetch of the appropriate texture / buffer
				vec3 dist = 1 - (offset + (1 - 2 * offset) * fractPos);
				float intMult = dist.x * dist.y * dist.z;//(1 - abs(cornerPos.x - vxPosOld.x)) * (1 - abs(cornerPos.y - vxPosOld.y)) * (1 - abs(cornerPos.z - vxPosOld.z));
				if (length(cornerLightData) > 0.001) {
					lightCol += intMult * cornerLightData.xyz;
					totalInt += intMult;
					#ifdef GI
						vec3 dist1 = 1 - (offset + (1 - 2 * offset) * fractPos1);
						float intMult1 = dist1.x * dist1.y * dist1.z;
						lightCol1 += intMult1 * cornerLightData.xyz;
						totalInt1 += intMult1;
					#endif
				}
			}
			lightCol /= totalInt;
			lightCol = 5 * log(0.2 * lightCol + 1);
			#ifdef GI
				lightCol1 /= totalInt1;
				lightCol1 = 5 * log(0.2 * lightCol1 + 1);
				vec3 dLightdn = clamp((1.0 / NORMAL_OFFSET) * (lightCol1 - lightCol), vec3(0), 0.3 * max(lightCol, lightCol1));
				#if ADVANCED_LIGHT_TRACING > 0
					return lightCol + 2 * GI_STRENGTH * dLightdn;
				#else
					return 2 * lightCol + 2 * GI_STRENGTH * dLightdn;
				#endif
			#else
				return 3 * lightCol;
			#endif
		}

		vec3 getBlockLight(vec3 vxPos, vec3 worldNormal, int mat, bool doScattering) {
			vec3 screenPos = gl_FragCoord.xyz / vec3(textureSize(colortex12, 0), 1);
			vec4 prevCol = texture2D(colortex12, screenPos.xy);

			return prevCol.xyz * 2;
		}

		vec3 getBlockLight(vec3 vxPos, vec3 normal, int mat) {
			return getBlockLight(vxPos, normal, mat, false);
		}
		vec3 getBlockLight(vec3 vxPos) {
			return getBlockLight(vxPos, vec3(0), 0);
		}
	#endif
#endif