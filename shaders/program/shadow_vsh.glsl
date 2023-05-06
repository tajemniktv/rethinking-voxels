#define VERTEX_SHADER
#include "/lib/common.glsl"

out vec2 texCoordV;
out vec2 lmCoordV;
out vec3 normalV;
out vec4 vertexColV;
out vec3 posV;
out vec4 positionV;
out vec3 blockCenterOffsetV;

flat out vec3 sunVecV, upVecV;
flat out int vertexID;
flat out int spriteSizeV;
flat out int matV;

in vec3 at_midBlock;
in vec4 mc_Entity;
in vec2 mc_midTexCoord;

uniform int entityId;
uniform int blockEntityId;
uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;
uniform ivec2 atlasSize;

#if defined WAVING_ANYTHING_TERRAIN || defined WAVING_WATER_VERTEX
	uniform float frameTimeCounter;

	uniform vec3 cameraPosition;
#endif

//Common Variables//
#if (defined WAVING_ANYTHING_TERRAIN || defined WAVING_WATER_VERTEX) && defined NO_WAVING_INDOORS
	vec2 lmCoord = vec2(0.0);
#endif

//Includes//
#include "/lib/util/spaceConversion.glsl"

#if defined WAVING_ANYTHING_TERRAIN || defined WAVING_WATER_VERTEX
	#include "/lib/materials/materialMethods/wavingBlocks.glsl"
#endif

void main() {
	matV = max(int(mc_Entity.x + 0.5), max(entityId, blockEntityId));

	texCoordV = (gl_TextureMatrix[0] * gl_MultiTexCoord0).xy;

	sunVecV = GetSunVector();
	upVecV = normalize(gbufferModelView[1].xyz);

	vec4 pos0 = shadowModelViewInverse * (shadowProjectionInverse * ftransform());

	posV = pos0.xyz / pos0.w;

	lmCoordV = GetLightMapCoordinates();
	
	#if defined WAVING_ANYTHING_TERRAIN || defined WAVING_WATER_VERTEX
		#ifdef NO_WAVING_INDOORS
			lmCoord = lmCoordV;
		#endif

		DoWave(pos0.xyz, matV);
	#endif

	#ifdef PERPENDICULAR_TWEAKS
		if (matV == 10004 || matV == 10016) { // Foliage
			vec2 midCoord = (gl_TextureMatrix[0] * vec4(mc_midTexCoord, 0, 1)).st;
			vec2 texMinMidCoord = texCoordV - midCoord;
			if (texMinMidCoord.y < 0.0) {
				vec3 normal = gl_NormalMatrix * gl_Normal;
				pos0.xyz += normal * 0.35;
			}
		}
	#endif

	blockCenterOffsetV = at_midBlock / 64.0;
	gl_Position = shadowProjection * (shadowModelView * pos0);
	float lVertexPos = length(gl_Position.xy);
	float distortFactor = lVertexPos * shadowMapBias + (1.0 - shadowMapBias);
	gl_Position.xy *= 1.0 / distortFactor;
	gl_Position.z = gl_Position.z * 0.2;

	vec2 spriteSize = atlasSize * abs(texCoordV - mc_midTexCoord);
	spriteSizeV = int(max(spriteSize.x, spriteSize.y) + 0.5);
	normalV = normalize(mat3(shadowModelViewInverse) * (gl_NormalMatrix * gl_Normal).xyz);
	vertexColV = gl_Color;
	vertexID = gl_VertexID;
}
