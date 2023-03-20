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
	//if (gl_FragCoord.y < 20 && gl_FragCoord.x < viewWidth / MAX_TRIS * numFaces) color = vec3(1);
	//else if (gl_FragCoord.x < 0.5 * viewWidth && gl_FragCoord.y < 0.5 * viewHeight) {
		vec4 dir;
		dir.xyz = vec3(-40);
		vec3 pos = fract(cameraPosition) + 10 - 0.01 * (gl_FragCoord.x - 0.5 * viewWidth) * vec3(1, 0, -1) - 0.004 * (gl_FragCoord.y - 0.5 * viewHeight) * vec3(1, -2, 1);
	//}
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
