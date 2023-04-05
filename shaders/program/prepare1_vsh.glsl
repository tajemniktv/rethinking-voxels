#include "/lib/common.glsl"

uniform mat4 gbufferModelView;
uniform mat4 gbufferProjection;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

#include "/lib/vx/SSBOs.glsl"

flat out mat4 reprojectionMatrix;

void main() {
	vec3 dcamPos = previousCameraPosition - cameraPosition;
	reprojectionMatrix =
		gbufferProjection *
		gbufferModelView *
		// the vec4s are interpreted as column vectors, not row vectors as suggested by this notation
		mat4(
			vec4(1, 0, 0, 0),
			vec4(0, 1, 0, 0),
			vec4(0, 0, 1, 0),
			vec4(dcamPos, 1)) * 
		gbufferPreviousModelViewInverse * 
		gbufferPreviousProjectionInverse;
	gl_Position = ftransform();
}