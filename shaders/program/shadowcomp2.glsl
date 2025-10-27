#include "/lib/common.glsl"

#ifdef CSH

#if SHADOW_QUALITY >= 1
    const ivec3 workGroups = ivec3(64, 64, 1);
#else
    #if SHADOW_QUALITY > 4 || SHADOW_SMOOTHING < 3
        const const ivec3 workGroups = ivec3(256, 256, 1);
    #else
        const const ivec3 workGroups = ivec3(128, 128, 1);
    #endif
#endif

layout(local_size_x = 32, local_size_y = 32) in;

#include "/lib/vx/voxelReading.glsl"

#include "/lib/materials/specificMaterials/translucents/interactiveWaterConsts.glsl"

layout(rgba16f) uniform image2D shadowcolorimg2;

#define gl_FragCoord (gl_GlobalInvocationID.xy + 0.5)
void main() {
    #ifdef INTERACTIVE_WATER
        // Optimized: Cache shadowMapResolution and half values
        vec2 shadowMapRes = vec2(shadowMapResolution);
        if (any(greaterThan(gl_FragCoord, shadowMapRes))) return;
        
        vec2 halfShadowMapRes = 0.5 * shadowMapRes;
        vec2 newFragCoord = mod(gl_FragCoord.xy, halfShadowMapRes);
        
        // Optimized: Use bit shift for multiply by 2
        int lodIndex = int(gl_FragCoord.x > halfShadowMapRes.x) +
                (int(gl_FragCoord.y > halfShadowMapRes.y) << 1);
        
        ivec3 floorCamPos = cameraPositionInt;
        ivec3 camOffset = cameraPositionInt - previousCameraPositionInt;
        if (cameraPositionInt.y == -98257195) {
            floorCamPos = ivec3(floor(cameraPosition) + 0.5 * sign(cameraPosition));
            camOffset = ivec3(1.1 * (floorCamPos - floor(previousCameraPosition)));
        }

        // Optimized: Cache quarter shadowMapResolution
        vec3 pos0 = vec3((newFragCoord.xy - 0.25 * shadowMapRes), 30).xzy;
        int waterHeight = -1000;
        bool hadAnyBlocks = false;
        
        // Optimized: Precomputed attenuation coefficients
        const vec4 attenuationCoeffs = vec4(0.994, 0.983, 0.95, 0.9);
        float attenuationCoeff = attenuationCoeffs[lodIndex];
        
        // Optimized: Cache voxel volume half sizes and bounds
        ivec2 voxelHalfSizeXZ = voxelVolumeSize.xz >> 1; // bit shift instead of divide
        int voxelHalfSizeY = voxelVolumeSize.y >> 1;
        vec2 voxelBoundsMin = -vec2(voxelHalfSizeXZ) + 5.0;
        vec2 voxelBoundsMax = vec2(voxelHalfSizeXZ) - 5.0;
        
        if (all(greaterThan(pos0.xz, voxelBoundsMin)) && all(lessThan(pos0.xz, voxelBoundsMax))) {
            // Optimized: Cache coord calculation base
            ivec2 coordBaseXZ = ivec2(pos0.xz) + voxelHalfSizeXZ;
            
            for (int k = voxelVolumeSize.y - 1; k >= 0; k--) {
                ivec3 coords = ivec3(coordBaseXZ.x, k, coordBaseXZ.y);
                int waterData = imageLoad(voxelCols, coords * ivec3(1, 2, 1)).r >> 26;
                if ((waterData & 1) == 1) {
                    waterHeight = coords.y - voxelHalfSizeY;
                    pos0.y = waterHeight + 0.5;
                    hadAnyBlocks = true;
                    break;
                }
                if (!hadAnyBlocks && (imageLoad(occupancyVolume, coords).r & 1) != 0) {
                    hadAnyBlocks = true;
                }
            }
        } else {
            attenuationCoeff = attenuationCoeff * 2.0 - 1.0;
        }
        if (!hadAnyBlocks) {
            waterHeight = WATER_DEFAULT_HEIGHT - 1 - floorCamPos.y;
        }
        waterHeight += 500;
        ivec2 prevCoords = ivec2(gl_FragCoord.xy) + camOffset.xz;
        vec4 data = vec4(0, 0, 0, waterHeight);

        if (waterHeight >= 0) {
            // Optimized: Fetch all 4 neighbors at once
            mat4 aroundData = mat4(
                texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2(-1, 0)),
                texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2( 1, 0)),
                texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2( 0,-1)),
                texelFetchOffset(shadowcolor2, prevCoords, 0, ivec2( 0, 1))
            );
            
            // Optimized: Cache camOffset.y and compute heights more efficiently
            int camOffsetY = camOffset.y;
            ivec4 aroundHeights = ivec4(transpose(aroundData)[3] + 1000.5) - (1000 + camOffsetY);
            
            // Optimized: Use vector comparison to zero out mismatched heights
            bvec4 heightMatch = equal(aroundHeights, ivec4(waterHeight));
            for (int i = 0; i < 4; i++) {
                if (!heightMatch[i]) {
                    aroundData[i].xyz = vec3(0);
                }
            }
            
            // Optimized: Unrolled inner loop and cached pow2 calculations
            for (int i = 0; i < 3; i++) {
                float waveDir0 = waveDirs[i][0];
                float waveDir1 = waveDirs[i][1];
                float waveDir0Sq = pow2(waveDir0);
                float waveDir1Sq = pow2(waveDir1);
                
                int idx0 = int(waveDir0 < 0.0);
                int idx1 = 2 + int(waveDir1 < 0.0);
                
                data[i] += waveDir0Sq * aroundData[idx0][i] + 
                           waveDir1Sq * aroundData[idx1][i];
            }
            
            // Optimized: Cache modulo and texture coordinate calculation
            const float INV_1024 = 0.0009765625; // 1/1024
            vec2 windCoord = pos0.xz + vec2(floorCamPos.xz % (5 << 10)); // bit shift instead of multiply
            vec3 wind = pow2(texture(shadowcolor3, mod(windCoord * 0.2, vec2(1024)) * INV_1024).xyz);
            
            // Optimized: Combined constant
            const float WIND_STRENGTH = 0.01 * WATER_BUMP_INTERACTIVE;
            data.xyz += WIND_STRENGTH * wind;
            data.xyz *= attenuationCoeff;
            
            // Optimized: Single clamp check instead of equal comparison
            if (!all(equal(data.xyz, clamp(data.xyz, 0.0, 100.0)))) data.xyz = vec3(0);
        }
        imageStore(shadowcolorimg2, ivec2(gl_GlobalInvocationID.xy), data);
    #endif
}

#endif
