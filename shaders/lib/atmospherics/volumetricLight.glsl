// Volumetric tracing from Robobo1221, highly modified

#include "/lib/colors/lightAndAmbientColors.glsl"
#include "/lib/vx/getLighting.glsl"


float GetDepth(float depth) {
	return 2.0 * near * far / (far + near - (2.0 * depth - 1.0) * (far - near));
}

float GetDistX(float dist) {
	return (far * (dist - near)) / (dist * (far - near));
}

vec4 DistortShadow(vec4 shadowpos, float distortFactor) {
	shadowpos.xy *= 1.0 / distortFactor;
	shadowpos.z = shadowpos.z * 0.2;
	shadowpos = shadowpos * 0.5 + 0.5;

	return shadowpos;
}

vec4 GetVolumetricLight(inout float vlFactor, vec3 translucentMult, float lViewPos, vec3 nViewPos, float VdotL, float VdotU, vec2 texCoord, float z0, float z1, float dither) {
	if (max(blindness, darknessFactor) > 0.1) return vec4(0.0);
	vec4 volumetricLight = vec4(0.0);
	vec3 volumetricBlockLight = vec3(0.0);
	vec3 vlColorReducer = vec3(1.0);

	#ifdef OVERWORLD
		vec3 vlColor = lightColor;

		float vlSceneIntensity = isEyeInWater != 1 ? vlFactor : 1.0;
		float vlMult = 1.0;

		if (sunVisibility < 0.5) {
			vlSceneIntensity = 0.0;
			vlMult = 0.6 + 0.4 * max0(far - lViewPos) / far;
			vlColor = normalize(pow(vlColor, vec3(1.0 - max0(1.0 - 1.5 * nightFactor))));
			vlColor *= 0.0766 + 0.0766 * vsBrightness;
		} else {
			vlColorReducer = 1.0 / sqrt(vlColor);
		}

		float VdotLM = max((VdotL + 1.0) / 2.0, 0.0);
		float VdotUM = mix(pow2(1.0 - max(VdotU, 0.0)), 1.0, 0.5 * vlSceneIntensity);
		      VdotUM = smoothstep1(VdotUM);
			  VdotUM = pow(VdotUM, min(lViewPos / far, 1.0) * (3.0 - 2.0 * vlSceneIntensity));
		vlMult *= mix(VdotUM * VdotLM, 0.5 + 0.5 * VdotLM, rainFactor2) * vlTime;
		vlMult *= mix(pow2(invNoonFactor) * 0.875 + 0.125, 1.0, max(vlSceneIntensity, rainFactor2));
		vlMult *= mix(0.25, 1.0, max(sunVisibility, invRainFactor));

		#if LIGHTSHAFT_QUALITY == 4
			int sampleCount = vlSceneIntensity < 0.5 ? 30 : 50;
		#elif LIGHTSHAFT_QUALITY == 3
			int sampleCount = vlSceneIntensity < 0.5 ? 15 : 30;
		#elif LIGHTSHAFT_QUALITY == 2
			int sampleCount = vlSceneIntensity < 0.5 ? 10 : 20;
		#elif LIGHTSHAFT_QUALITY == 1
			int sampleCount = vlSceneIntensity < 0.5 ? 6 : 12;
		#endif
		#ifndef TAA
			sampleCount *= 2;
		#endif
	#else
		float vlSceneIntensity = 0.0;

		#if LIGHTSHAFT_QUALITY == 4
			int sampleCount = 20;
		#elif LIGHTSHAFT_QUALITY == 3
			int sampleCount = 16;
		#elif LIGHTSHAFT_QUALITY == 2
			int sampleCount = 12;
		#elif LIGHTSHAFT_QUALITY == 1
			int sampleCount = 10;
		#elif LIGHTSHAFT_QUALITY == 0
			int sampleCount = 8;
		#endif
	#endif

	float addition = 1.0;
	float maxDist = mix(max(far, 96.0) * 0.55, 80.0, vlSceneIntensity);
	float distMult = maxDist / (sampleCount + addition);
	float sampleMultIntense = isEyeInWater != 1 ? 1.0 : 0.85;

	float depth0 = GetDepth(z0);
	float depth1 = GetDepth(z1);
	#ifdef END
		if (z0 == 1.0) depth0 = 1000.0;
		if (z1 == 1.0) depth1 = 1000.0;
	#endif

	// Fast but inaccurate perspective distortion approximation
	float viewFactor = 1.0 - 0.7 * pow2(dot(nViewPos.xy, nViewPos.xy));
	maxDist *= viewFactor;
	distMult *= viewFactor;
	
	#ifdef OVERWORLD
		float maxCurrentDist = min(depth1, maxDist);
	#else
		float maxCurrentDist = min(depth1, far);
	#endif

	for (int i = 0; i < sampleCount; i++) {
		float currentDist = (i + dither) * distMult + addition;

		if (currentDist > maxCurrentDist) break;

		vec4 viewPos = gbufferProjectionInverse * (vec4(texCoord, GetDistX(currentDist), 1.0) * 2.0 - 1.0);
		viewPos /= viewPos.w;
		vec4 wpos = gbufferModelViewInverse * viewPos;
		vec3 playerPos = wpos.xyz / wpos.w;
		#ifdef END
			vec4 enderBeamSample = vec4(DrawEnderBeams(VdotU, playerPos), 1.0);
			enderBeamSample /= sampleCount;
		#endif
		float blSampleMult = 1.0 / sampleCount;
		#ifdef OVERWORLD
			float percentComplete = currentDist / maxDist;
			float sampleMult = mix(percentComplete * 3.0, sampleMultIntense, max(rainFactor, vlSceneIntensity));
			if (currentDist < 5.0) sampleMult *= smoothstep1(clamp(currentDist / 5.0, 0.0, 1.0));
			sampleMult /= sampleCount;
		#elif defined NETHER
			blSampleMult *= VBL_NETHER_MULT;
		#else
			blSampleMult *= VBL_END_MULT;
		#endif

		vec3 blSample = vec3(0.0);
		vec3 vxPos = getVxPos(playerPos);
		if (isInRange(vxPos, 2)) {
			blSample = getBlockLight(vxPos);
		}
		#ifdef REALTIME_SHADOWS
			float shadowSample = 1.0;
			vec3 vlSample = vec3(1.0);
			vec3 prevVxPos = getPreviousVxPos(playerPos);
			if (isInRange(prevVxPos, 2)) {
				vlSample = getSunLight(prevVxPos, true);
			#ifndef END
				} else {
					vlSample = vec3(eyeBrightnessSmooth.y / 240.0);
			#endif
			}
			vlSample *= vlSample + 0.1;
			shadowSample = dot(vlSample, vec3(1)) > 0.5 ? 1.0 : 0.0;
		#endif
		if (currentDist > depth0)  {
			#ifdef REALTIME_SHADOWS
				vlSample *= translucentMult;
			#endif
			blSample *= translucentMult;
		}
		volumetricBlockLight += blSample * blSampleMult;

		#ifdef REALTIME_SHADOWS
			#ifdef OVERWORLD
				volumetricLight += vec4(vlSample, shadowSample) * sampleMult;
			#elif defined END
				volumetricLight += vec4(vlSample, shadowSample) * enderBeamSample;
			#endif
		#endif
	}

	#if defined OVERWORLD && LIGHTSHAFT_BEHAVIOUR == 1 && defined REALTIME_SHADOWS
		if (viewWidth + viewHeight - gl_FragCoord.x - gl_FragCoord.y < 1.5) {
			if (frameCounter % int(0.06666 / frameTimeSmooth + 0.5) == 0) { // Change speed is not too different above 10 fps
				if (eyeBrightness.y < 180) {
					vec4 wpos = vec4(shadowModelView[3][0], shadowModelView[3][1], shadowModelView[3][2], shadowModelView[3][3]);
					wpos = shadowModelViewInverse * wpos;
					wpos /= wpos.w;
					float shadowSample = length(getSunLight(getPreviousVxPos(wpos.xyz))) > 0.3 ? 1.0 : 0.0;
					if (shadowSample < 0.5) {
						int salsX = 8;
						int salsY = 5;
						vec2 viewM = view / vec2(salsX, salsY);
						float skySample = 0.0;
						for (float i = 0.5; i < salsX; i++) {
							for (float h = 0.9; h < salsY; h++) {
								skySample += float(texelFetch(depthtex0, ivec2(viewM * vec2(i, h)), 0).r == 1.0);
							}
						}
						if (skySample < 1.5) {
							vlFactor = min(vlFactor + OSIEBCA*2, 1.0);
						} else vlFactor = max(vlFactor - OSIEBCA*3, 0.0);
					} else vlFactor = max(vlFactor - OSIEBCA*3, 0.0);
				} else vlFactor = max(vlFactor - OSIEBCA*3, 0.0);
			}
		} else vlFactor = 0.0;

		/*for (float i = 0.5; i < salsX; i++) { // Show Scene Aware check positions
			for (float h = 0.9; h < salsY; h++) {
				vec2 dis = abs(viewM * vec2(i, h) - gl_FragCoord.xy);
				if (dis.x + dis.y < 10.0) return vec4(1.0);
			}
		}*/
	#endif

	#ifdef OVERWORLD
		volumetricLight.rgb *= vlMult * pow(vlColor, vec3(0.5 + 0.5 * mix(invNoonFactor, (1.0 + sunFactor), rainFactor)));
	#endif

	volumetricLight.rgb += BLOCKLIGHT_SHAFT_STRENGTH * volumetricBlockLight;	

	volumetricLight = max(volumetricLight, vec4(0.0));
	volumetricLight.a = min(volumetricLight.a, 1.0);

	return volumetricLight;
}
