#include "/lib/common.glsl"

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousModelView;
uniform mat4 gbufferPreviousProjection;
uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

flat out mat4 reprojectionMatrix;

void main() {
	vec3 dcamPos = cameraPosition - previousCameraPosition;
	reprojectionMatrix =
		gbufferPreviousProjection *
		gbufferPreviousModelView *
		// the vec4s are interpreted as column vectors, not row vectors as suggested by this notation
		mat4(
			vec4(1, 0, 0, 0),
			vec4(0, 1, 0, 0),
			vec4(0, 0, 1, 0),
			vec4(dcamPos, 1)) *
		gbufferModelViewInverse *
		gbufferProjectionInverse;
	gl_Position = ftransform();
}