#include "/lib/common.glsl"

#if BL_SHADOW_MODE == 1
flat in vec3 sunDir;
in vec2 texCoord;

uniform float viewWidth;
uniform float viewHeight;
vec2 view = vec2(viewWidth, viewHeight);

uniform int frameCounter;

uniform ivec2 atlasSize;

uniform sampler2D colortex1;
uniform sampler2D colortex4;
uniform sampler2D colortex5;

uniform sampler2D colortex15;
#define ATLASTEX colortex15

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferModelViewInverse;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

const bool colortex15MipmapEnabled = false;

ivec2 offsets[4] = ivec2[4](ivec2(0, 0), ivec2(0, 1), ivec2(1, 1), ivec2(1, 0));

#include "/lib/vx/voxelMapping.glsl"
#include "/lib/vx/voxelReading.glsl"
#define PP_BL_SHADOWS
#define FF_IS_UPDATED
#define VX_NORMAL_MARGIN 0.08
#include "/lib/vx/getLighting.glsl"
#endif
void main() {
    #if BL_SHADOW_MODE == 1
    float tex4val = texelFetch(colortex4, ivec2(gl_FragCoord.xy), 0).a;
    ivec2 pixelCoord = ivec2(gl_FragCoord.xy) * 2 + offsets[frameCounter%4];
    float depth0 = texelFetch(depthtex0, pixelCoord, 0).r;
    float depth1 = texelFetch(depthtex1, pixelCoord, 0).r;
    vec4 pos0 = vec4((pixelCoord + 0.5) / view, depth1, 1) * 2 - 1;
    vec4 viewPos = gbufferProjectionInverse * pos0;
    vec4 playerPos = gbufferModelViewInverse * viewPos;
    playerPos /= playerPos.w;
    vec3 vxPos = playerPos.xyz + fract(cameraPosition);
    vec3 normal = int(texelFetch(colortex1, pixelCoord, 0).g * 255.1) != 4 || depth0 < 0.56? texelFetch(colortex4, pixelCoord, 0).rgb * 2 - 1 : vec3(0);
    vec3 blockLight = getBlockLight(vxPos + 0.02 * normal, normal, 0) * 0.5;

    /*RENDERTARGETS:4*/
    gl_FragData[0] = vec4(blockLight, tex4val);
    #endif
}