#ifndef IRRADIANCECACHE
#define IRRADIANCECACHE

// Precompute reciprocal to avoid division (set this uniform in your code)
// If voxelVolumeSize is already a uniform, add: uniform vec3 rcpVoxelVolumeSize = 1.0 / voxelVolumeSize;
#ifndef rcpVoxelVolumeSize
    #define rcpVoxelVolumeSize (1.0 / voxelVolumeSize)
#endif

// Inlined and optimized bounds check
#define IS_IN_RANGE(vxPos) (all(greaterThan(vxPos, -0.5*voxelVolumeSize)) && all(lessThan(vxPos, 0.5*voxelVolumeSize)))

uniform sampler3D irradianceCache;

vec3 readIrradianceCache(vec3 vxPos, vec3 normal) {
    // Early exit with inlined bounds check
    if (!IS_IN_RANGE(vxPos)) return vec3(0);

    // Optimized: use multiplication instead of division and combine operations
    vxPos = (vxPos + 0.5 * normal) * rcpVoxelVolumeSize;
    vxPos = (vxPos + 0.5) * vec3(1.0, 0.5, 1.0);

    vec4 color = textureLod(irradianceCache, vxPos, 0);
    // Use reciprocal multiplication instead of division
    return color.rgb * (1.0 / max(color.a, 0.0001));
}

vec3 readSurfaceVoxelBlocklight(vec3 vxPos, vec3 normal) {
    if (!IS_IN_RANGE(vxPos)) return vec3(0);

    vxPos = (vxPos + 0.5 * normal) * rcpVoxelVolumeSize;
    vxPos = (vxPos + vec3(0.5, 1.5, 0.5)) * vec3(1.0, 0.5, 1.0);

    vec4 color = textureLod(irradianceCache, vxPos, 0);
    float lColor = length(color.rgb);
    // Optimized: compute reciprocal once and check combined with multiplication
    if (lColor > 0.01) {
        float rcpLColor = 1.0 / lColor;
        color.rgb *= log(lColor + 1.0) * rcpLColor;
    }
    return color.rgb;
}

vec3 readVolumetricBlocklight(vec3 vxPos) {
    if (!IS_IN_RANGE(vxPos)) return vec3(0);

    vxPos = vxPos * rcpVoxelVolumeSize;
    vxPos = (vxPos + vec3(0.5, 1.5, 0.5)) * vec3(1.0, 0.5, 1.0);

    vec4 color = textureLod(irradianceCache, vxPos, 0);
    return color.rgb * (1.0 / max(color.a, 0.0001));
}
#endif