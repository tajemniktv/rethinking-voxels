#include "/lib/util/reprojection.glsl"

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
		float depthCheck = texelFetch(depthtex1, texelCoord + neighbourhoodOffsets[i], 0).r;
		if (abs(min(0.5, GetLinearDepth(depthCheck)) - lindepth) * (0.5 / lindepth + 1) > 0.05) edge = 10.0;
		vec3 clr = texelFetch(colortex3, texelCoord + neighbourhoodOffsets[i], 0).rgb;
		minclr = min(minclr, clr); maxclr = max(maxclr, clr);
	}

	tempColor = mix(tempColor, clamp(tempColor, minclr, maxclr), 0.5);
}

void DoTAA(inout vec3 color, inout vec4 temp) {
	if (int(texelFetch(colortex1, texelCoord, 0).g * 255.1) == 4) { // No SSAO, No TAA
		return;
	}

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
	vec2 velocity = (texCoord - prvCoord.xy) * view;

	float blendFactor = float(prvCoord.x > 0.0 && prvCoord.x < 1.0 &&
	                          prvCoord.y > 0.0 && prvCoord.y < 1.0);
	//float blendMinimum = 0.6;
	//float blendVariable = 0.5;
	//float blendConstant = 0.4;
	float blendMinimum = 0.01;
	float blendVariable = 0.28;
	float blendConstant = 0.65;
	float lengthVelocity = (10 * edge + 1) * length(velocity);
	float lPrvDepth0 = GetLinearDepth(prvCoord.z);
	float lPrvDepth1 = GetLinearDepth(tempColor.w);
	float ddepth = abs(lPrvDepth0 - lPrvDepth1) * (1 / lPrvDepth0 + 1);// / (lPrvDepth0 + lPrvDepth1);
	blendFactor *= max(exp(-lengthVelocity) * blendVariable + blendConstant - ddepth * (10 - edge) - length(cameraOffset) * edge, blendMinimum);
	
	color = mix(color, tempColor.xyz, blendFactor);
	temp = vec4(color, depth);
	//if (edge > 0.05) color.b = 1.0;
	//if (ddepth > 0.02) color.r = 1.0;
}