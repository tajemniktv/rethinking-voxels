#include "/lib/common.glsl"

uniform sampler2D colortex8;
uniform float viewWidth;
uniform float viewHeight;
ivec2 view = ivec2(viewWidth + 0.5, viewHeight + 0.5);
#define BLUR_SIZE_0 3

void main() {
	#ifdef DENOISING
		ivec2 coord = ivec2(gl_FragCoord.xy);
		vec4 outColor = texelFetch(colortex8, coord, 0);
		if (all(lessThan(coord, view / 2))) {
			ivec2[2] blurBox = ivec2[2](
				max(ivec2(-BLUR_SIZE_0), -coord),
				min(ivec2(BLUR_SIZE_0), view / 2 - coord) + 1
			);
			vec3[2] grad = vec3[2](vec3(0), vec3(0));
			vec3 avg = vec3(0);
			vec3[3] H = vec3[3](vec3(0), vec3(0), vec3(0));
			for (int x = blurBox[0].x; x < blurBox[1].x; x++) {
				for (int y = blurBox[0].y; y < blurBox[1].y; y++) {
					vec3 col = texelFetch(colortex8, coord + ivec2(x, y), 0).rgb;
					avg += col;
					grad[0] += col * x;
					grad[1] += col * y;
				}
			}
			avg /= (blurBox[1].x - blurBox[0].x) * (blurBox[1].y - blurBox[0].y);
			vec2 gradWeightAccum = (
				blurBox[0] * (blurBox[0] - 1) / 2 +
				blurBox[1] * (blurBox[1] - 1) / 2
			) * (blurBox[1] - blurBox[0]).yx;
			grad[0] /= max(gradWeightAccum[0], 0.00001);
			grad[1] /= max(gradWeightAccum[1], 0.00001);
			float gradVar = length(dFdx(grad[1])) + length(dFdy(grad[0]));
			grad[0] /= max(avg, 0.00001);
			grad[1] /= max(avg, 0.00001);
			vec3 res = vec3(0);
			vec3 weight = vec3(0.0000001);
			for (int x = blurBox[0].x; x < blurBox[1].x; x++) {
				for (int y = blurBox[0].y; y < blurBox[1].y; y++) {
					vec3 col = texelFetch(colortex8, coord + ivec2(x, y), 0).rgb;
					vec3 thisWeight;
					for (int i = 0; i < 3; i++) thisWeight[i] = max(0, 1 - gradVar * length(vec2(x, y)) - 2 * abs(dot(vec2(x, y), vec2(grad[0][i], grad[1][i]))));
					res += col * thisWeight;
					weight += thisWeight;
				}
			}
			outColor.xyz = res / weight;
		}
		/*RENDERTARGETS:8*/
		gl_FragData[0] = outColor;
	#endif
}