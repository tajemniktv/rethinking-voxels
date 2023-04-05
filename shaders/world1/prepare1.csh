#version 430

layout(local_size_x = 1) in;

layout(rgba16f) uniform image2D colorimg0;

void main() {
	imageStore(colorimg0, ivec2(gl_GlobalInvocationID.xy), vec4(2.0));
}