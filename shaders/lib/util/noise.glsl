#ifdef WATERHEIGHT
	float GetWaterHeightMap(vec2 waterPos, vec3 nViewPos, vec2 wind) {
		float waveNoise = 0;
		vec2 noiseA = 0.5 - texture2D(noisetex, waterPos - wind * 0.6).rg;
		vec2 noiseB = 0.5 - texture2D(noisetex, waterPos * 2.0 + wind).rg;
		waveNoise = noiseA.r - noiseA.r * noiseB.r + noiseB.r * 0.6 + (noiseA.g + noiseB.g) * 2.5;
		return waveNoise;
	}
#endif
#ifdef HASH33
	vec3 hash33(vec3 p3) {
		p3 = fract(p3 * vec3(.1031, .1030, .0973));
		p3 += dot(p3, p3.yxz+33.33);
		return fract((p3.xxy + p3.yxx)*p3.zyx);
	}
#endif
