const float regularEdge = 20.0;
const float extraEdgeMult = 3.0;

// Previous frame reprojection from Chocapic13
vec2 Reprojection(vec3 pos, vec3 cameraOffset) {
	pos = pos * 2.0 - 1.0;

	vec4 viewPosPrev = gbufferProjectionInverse * vec4(pos, 1.0);
	viewPosPrev /= viewPosPrev.w;
	viewPosPrev = gbufferModelViewInverse * viewPosPrev;

	vec4 previousPosition = viewPosPrev + vec4(cameraOffset, 0.0);
	previousPosition = gbufferPreviousModelView * previousPosition;
	previousPosition = gbufferPreviousProjection * previousPosition;
	return previousPosition.xy / previousPosition.w * 0.5 + 0.5;
}

vec3 Reprojection3D(vec3 pos, vec3 cameraOffset) {
	pos = pos * 2.0 - 1.0;

	vec4 viewPosPrev = gbufferProjectionInverse * vec4(pos, 1.0);
	viewPosPrev /= viewPosPrev.w;
	viewPosPrev = gbufferModelViewInverse * viewPosPrev;

	vec4 previousPosition = viewPosPrev + vec4(cameraOffset, 0.0);
	previousPosition = gbufferPreviousModelView * previousPosition;
	previousPosition = gbufferPreviousProjection * previousPosition;
	return previousPosition.xyz / previousPosition.w * 0.5 + 0.5;
}

ivec2 neighbourhoodOffsets[8] = ivec2[8](
	ivec2(-1, -1),
	ivec2( 0, -1),
	ivec2( 1, -1),
	ivec2(-1,  0),
	ivec2( 1,  0),
	ivec2(-1,  1),
	ivec2( 0,  1),
	ivec2( 1,  1)
);

void NeighbourhoodClamping(vec3 color, inout vec3 tempColor, float depth, inout float edge) {
	vec3 minclr = color, maxclr = color;
	float lindepth = min(0.5, GetLinearDepth(depth));
	for (int i = 0; i < 8; i++) {
		ivec2 texelCoordM = texelCoord + neighbourhoodOffsets[i];

		float depthCheck = texelFetch(depthtex1, texelCoordM, 0).r;
		if (abs(min(0.5, GetLinearDepth(depthCheck)) - lindepth) * (0.5 / lindepth + 1) > 0.05) {
			edge = regularEdge;

			if (int(texelFetch(colortex1, texelCoordM, 0).g * 255.1) == 253) // Reduced Edge TAA
				edge *= extraEdgeMult;
		}

		vec3 clr = texelFetch(colortex3, texelCoordM, 0).rgb;
		minclr = min(minclr, clr); maxclr = max(maxclr, clr);
	}
	#if defined PP_BL_SHADOWS || defined PP_SUN_SHADOWS
	tempColor = mix(tempColor, clamp(tempColor, minclr, maxclr), 0.5);
	#else
	tempColor = clamp(tempColor, minclr, maxclr);
	#endif
}

void DoTAA(inout vec3 color, inout vec4 temp) {
	int materialMask = int(texelFetch(colortex1, texelCoord, 0).g * 255.1);

	if (materialMask == 254) // No SSAO, No TAA
		return;

	float depth = texelFetch(depthtex1, texelCoord, 0).r;
	vec3 coord = vec3(texCoord, depth);
	vec3 cameraOffset = cameraPosition - previousCameraPosition;
	vec3 prvCoord = Reprojection3D(coord, cameraOffset);
	
	vec2 view = vec2(viewWidth, viewHeight);
	vec4 tempColor = texture2D(colortex2, prvCoord.xy);
	if (tempColor.xyz == vec3(0.0)) { // Fixes the first frame
		temp = vec4(color, depth);
		return;
	}

	float edge = 0.0;
	NeighbourhoodClamping(color, tempColor.xyz, depth, edge);

	if (materialMask == 253) // Reduced Edge TAA
		edge *= extraEdgeMult;

	vec2 velocity = (texCoord - prvCoord.xy) * view;
	float blendFactor = float(prvCoord.x > 0.0 && prvCoord.x < 1.0 &&
	                          prvCoord.y > 0.0 && prvCoord.y < 1.0);
	//float blendMinimum = 0.6;
	//float blendVariable = 0.5;
	//float blendConstant = 0.4;
	float blendConstant = 0.65;
	#if defined PP_SUN_SHADOWS || defined PP_BL_SHADOWS
	float blendMinimum = 0.01;
	float blendVariable = 0.28;
	float velocityFactor = dot(velocity, velocity) * 10.0;
	float lPrvDepth0 = GetLinearDepth(prvCoord.z);
	float lPrvDepth1 = GetLinearDepth(tempColor.w);
	float ddepth = abs(lPrvDepth0 - lPrvDepth1) * (1 / abs(lPrvDepth0) + 1);
	#else
	float blendMinimum = 0.3;
	float blendVariable = 0.25;
	float velocityFactor = dot(velocity, velocity) * 10.0;

	float ddepth = 0;
	#endif
	blendFactor *= max(exp(-velocityFactor) * blendVariable + blendConstant - ddepth * (edge < 0.1 ? 10 : 0) - length(cameraOffset) * edge, blendMinimum);
	
	color = mix(color, tempColor.xyz, blendFactor);
	temp = vec4(color, depth);
}
