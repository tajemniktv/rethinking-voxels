////////////////////////////////////////
// Complementary Reimagined by EminGT //
////////////////////////////////////////

//Common//
#include "/lib/common.glsl"

//////////Fragment Shader//////////Fragment Shader//////////Fragment Shader//////////
#ifdef FRAGMENT_SHADER

noperspective in vec2 texCoord;

//Uniforms//
uniform float viewWidth, viewHeight;

uniform sampler2D colortex3;

//SSBOs//
#include "/lib/vx/SSBOs.glsl"

//Pipeline Constants//
#ifndef TAA
	const bool colortex3MipmapEnabled = true;
#endif

//Common Variables//

//Common Functions//

//Includes//
#ifdef FXAA
	#include "/lib/antialiasing/fxaa.glsl"
#endif

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
ivec2 atlasSize = ivec2(1);
uniform sampler2D colortex15;

#include "/lib/vx/raytrace.glsl"

//Program//
void main() {
	vec3 color = texelFetch(colortex3, texelCoord, 0).rgb;

	#ifdef FXAA
		FXAA311(color);
	#endif

	if (false && all(lessThan(abs(texelCoord - ivec2(viewWidth, viewHeight) / 2), ivec2(viewWidth, viewHeight) / 4))) {
		vec4 playerPos = vec4(gl_FragCoord.xy / vec2(viewWidth, viewHeight) * 2 - 1, 0.999, 1);
		playerPos = gbufferProjectionInverse * playerPos;
		playerPos = gbufferModelViewInverse * playerPos;
		playerPos.xyz = 30 * normalize(playerPos.xyz);
		ray_hit_t rayHit = betterRayTrace(fract(cameraPosition), playerPos.xyz, colortex15);
		color = mix(color, rayHit.rayColor.rgb, 0.5);
	}
	if (texelCoord.y < 10) {
		if (texelCoord.x < numFaces * viewWidth / float(MAX_TRIS)) color = 0.1 * lights[texelCoord.x / 2].pos + 0.5;
	}

	/*DRAWBUFFERS:3*/
	gl_FragData[0] = vec4(color, 1.0);
}

#endif

//////////Vertex Shader//////////Vertex Shader//////////Vertex Shader//////////
#ifdef VERTEX_SHADER

noperspective out vec2 texCoord;

//Uniforms//

//Attributes//

//Common Variables//

//Common Functions//

//Includes//

//Program//
void main() {
	gl_Position = ftransform();

	texCoord = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;
}

#endif
