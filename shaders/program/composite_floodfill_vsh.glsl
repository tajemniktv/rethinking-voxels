#include "/lib/common.glsl"

out vec2 texCoord;

#if defined GI && defined REALTIME_SHADOWS
out vec3 sunVec;
#endif

uniform sampler2D colortex8;

vec3 getWorldSunVector() {
    const vec2 sunRotationData = vec2(cos(sunPathRotation * 0.01745329251994), -sin(sunPathRotation * 0.01745329251994));
    #ifdef OVERWORLD
        float ang = fract(timeAngle - 0.25);
        ang = (ang + (cos(ang * 3.14159265358979) * -0.5 + 0.5 - ang) / 3.0) * 6.28318530717959;
        return vec3(-sin(ang), cos(ang) * sunRotationData) + vec3(0.00001);
    #elif defined END
        return vec3(0.0, sunRotationData) + vec3(0.00001);
    #else
        return vec3(0.0);
    #endif
}

void main() {
    gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * gl_Vertex);
    gl_Position.xy = ((0.5 * gl_Position.xy + 0.5) * (shadowMapResolution - vec2(1, 0)) + vec2(1, 0)) / vec2(textureSize(colortex8, 0)) * 2.0 - 1.0;
    texCoord = 0.5 * gl_Position.xy + 0.5;
}
