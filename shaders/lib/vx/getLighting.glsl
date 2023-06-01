#ifndef LIGHTING
	#define LIGHTING
	#ifndef COLORTEX12
		#define COLORTEX12
		uniform sampler2D colortex12;
	#endif
	#include "/lib/vx/voxelMapping.glsl"
	#include "/lib/vx/SSBOs.glsl"
	#define DEBUG_OCCLUDERS
		vec3 getGI(vec3 vxPos, vec3 normal, int mat, bool doScattering) {
		vxPos += 0.5 * normal;
		vec3 lightCol = vec3(0);
		vec3 vxPosOld = vxPos + 8 * (floor(0.125 * cameraPosition) - floor(0.125 * previousCameraPosition)) - 0.5;
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
	#ifdef PER_BLOCK_LIGHT
		vec3 getBlockLight(vec3 vxPos, vec3 worldNormal, int mat, bool doScattering) {
			if (length(worldNormal) > 0.01)
				return readIrradianceCache(vxPos + 0.5 * worldNormal, worldNormal);
			else
				return readIrradianceCache(vxPos, 6);
		}
	#else
		vec3 getBlockLight(vec3 vxPos, vec3 worldNormal, int mat, bool doScattering) {
			vec3 screenPos = gl_FragCoord.xyz / vec3(textureSize(colortex12, 0), 1);
			vec4 prevCol = texture2D(colortex12, screenPos.xy);

			return prevCol.xyz * 2;
		}
	#endif

	vec3 getBlockLight(vec3 vxPos, vec3 normal, int mat) {
		return getBlockLight(vxPos, normal, mat, false);
	}
	vec3 getBlockLight(vec3 vxPos) {
		return getBlockLight(vxPos, vec3(0), 0, false);
	}
#endif
