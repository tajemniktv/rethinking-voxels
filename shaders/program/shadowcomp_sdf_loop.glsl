{
    if (gl_LocalInvocationIndex == 0u) {
        isActive = (gl_WorkGroupID.x + gl_WorkGroupID.y + gl_WorkGroupID.z + frameCounter) % max(1, SDF_UPDATE_INTERVAL);
    }
    barrier();
    memoryBarrierShared();
    
    // Optimized: Precompute constants
    const float INV_SQRT3 = 0.57735027; // 1/sqrt(3)
    float distScale = 1.0 / float(1 << j); // Cache bit shift division
    float invDistScale = float(1 << j);
    int frameModulo2 = frameCounter & 1; // Bit AND instead of modulo
    int yOffsetBase = (frameModulo2 * 2 + (j >> 2)) * voxelVolumeSize.y; // >> 2 instead of /4
    
    for (int k = 0; k < 2; k++) {
        int index = int(gl_LocalInvocationIndex + k * 512);
        if (index > 1000) break;
        
        // Optimized: Cache modulo results
        int indexMod10 = index % 10;
        int indexDiv10 = index / 10;
        ivec3 currentLocalCoord = ivec3(indexMod10, indexDiv10 % 10, indexDiv10 / 10) - 1;
        
        ivec3 texCoord = currentLocalCoord + baseCoord;
        int thisLocalOccupancy = imageLoad(occupancyVolume, texCoord).r;
        ivec3 prevTexCoord0 = texCoord + (1<<j) * floorCamPosOffset;
        ivec3 prevTexCoord = prevTexCoord0 + ivec3(0, yOffsetBase, 0);
        
        if ((thisLocalOccupancy >> j & 1) == 1) {
            fullDist[currentLocalCoord.x+1][currentLocalCoord.y+1][currentLocalCoord.z+1] = (1.0 - INV_SQRT3) * distScale;
        } else if (
            all(greaterThanEqual(prevTexCoord0, ivec3(0))) &&
            all(lessThan(prevTexCoord0, voxelVolumeSize))
        ) {
            fullDist[currentLocalCoord.x+1][currentLocalCoord.y+1][currentLocalCoord.z+1] = imageLoad(distanceFieldI, prevTexCoord)[(j & 3)] + distScale;
        } else {
            atomicExchange(isActive, 0);
            #if j > 0
                float prevDist = 10000.0;
                ivec3 prevCoord = (prevTexCoord0 >> 1) + (voxelVolumeSize >> 2) + ivec3(0, (frameModulo2 * 2 + ((j-1) >> 2)) * voxelVolumeSize.y, 0);
                ivec3 prevFractCoord = prevTexCoord0 & 1; // Bit AND instead of modulo
                
                // Optimized: Precompute distOffsets
                float distScalePrev = 1.0 / float(1 << (j-1));
                const float distOffset0 = 0.25;
                const float distOffset1 = 0.75;
                int jMinus1Mod4 = (j - 1) & 3;
                
                for (int n = 0; n < 8; n++) {
                    // Optimized: Use bit operations for offset calculation
                    ivec3 offset = ivec3(n & 1, (n >> 1) & 1, (n >> 2) & 1) * ((prevFractCoord << 1) - 1);
                    float distOffset = (n == 0 ? distOffset0 : distOffset1) * distScalePrev;
                    prevDist = min(prevDist, distOffset + imageLoad(distanceFieldI, prevCoord + offset)[jMinus1Mod4]);
                }
                fullDist[currentLocalCoord.x+1][currentLocalCoord.y+1][currentLocalCoord.z+1] = prevDist + distScale;
            #else
                fullDist[currentLocalCoord.x+1][currentLocalCoord.y+1][currentLocalCoord.z+1] = 1000.0;
            #endif
        }
    }
    barrier();
    memoryBarrierShared();

    theseDists[j] = (thisOccupancy >> j & 1) == 1 ? -INV_SQRT3 * distScale : 1000;
    if (isActive == 0) {
        #if j > 0
            ivec3 prevTexCoord0 = texCoord + (1<<j) * floorCamPosOffset;
            ivec3 prevCoord = (prevTexCoord0 >> 1) + (voxelVolumeSize >> 2) + ivec3(0, (frameModulo2 * 2 + ((j-1) >> 2)) * voxelVolumeSize.y, 0);
            ivec3 prevFractCoord = prevTexCoord0 & 1;

            float prevDist = 10000.0;
            float distScalePrev = 1.0 / float(1 << (j-1));
            int jMinus1Mod4 = (j - 1) & 3;
            
            for (int n = 0; n < 8; n++) {
                ivec3 offset = ivec3(n & 1, (n >> 1) & 1, (n >> 2) & 1) * ((prevFractCoord << 1) - 1);
                float distOffset = (n == 0 ? 0.25 : 0.75) * distScalePrev;
                prevDist = min(prevDist, distOffset + imageLoad(distanceFieldI, prevCoord + offset)[jMinus1Mod4]);
            }
            
            // Optimized: Cache threshold calculations
            float levelFadeThreshold = levelFadeDist * 1.5 * distScale;
            if (prevDist < levelFadeThreshold) {
            #endif

            // Optimized: Precomputed offset lookup table for 27 neighbors
            const ivec3 neighborOffsets[27] = ivec3[27](
                ivec3(0,0,0), ivec3(1,0,0), ivec3(2,0,0),
                ivec3(0,1,0), ivec3(1,1,0), ivec3(2,1,0),
                ivec3(0,2,0), ivec3(1,2,0), ivec3(2,2,0),
                ivec3(0,0,1), ivec3(1,0,1), ivec3(2,0,1),
                ivec3(0,1,1), ivec3(1,1,1), ivec3(2,1,1),
                ivec3(0,2,1), ivec3(1,2,1), ivec3(2,2,1),
                ivec3(0,0,2), ivec3(1,0,2), ivec3(2,0,2),
                ivec3(0,1,2), ivec3(1,1,2), ivec3(2,1,2),
                ivec3(0,2,2), ivec3(1,2,2), ivec3(2,2,2)
            );
            
            for (int n = 0; n < 27; n++) {
                ivec3 c2 = localCoord + neighborOffsets[n];
                theseDists[j] = min(theseDists[j], fullDist[c2.x][c2.y][c2.z]);
                #if SDF_UPDATE_INTERVAL == 0
                    ivec3 c3 = localCoord + (neighborOffsets[n] << 1) - 1; // Bit shift instead of multiply
                    theseDists[j] = min(
                        theseDists[j],
                        all(greaterThanEqual(c3, ivec3(0))) &&
                        all(lessThan(c3, ivec3(10))) ?
                        fullDist[c3.x][c3.y][c3.z] + distScale : 1000
                    );
                #endif
            }
            #if j > 0
                float levelFadeThresholdLow = levelFadeDist * distScale;
                if (prevDist > levelFadeThresholdLow) {
                    theseDists[j] = mix(theseDists[j], prevDist, prevDist * invDistScale / (0.5 * levelFadeDist) - 2.0);
                }
            } else {
                theseDists[j] = prevDist;
            }
            const float edgeDist = 4.0;
            const float invEdgeDist = 0.25; // 1/4
            float edgeDecider = min(infnorm(max(abs(texCoord + 0.5 - 0.5 * voxelVolumeSize) + edgeDist + 0.5 - 0.5 * voxelVolumeSize, 0.0)) * invEdgeDist, 1.0);
            if (edgeDecider > 0.01) {
                theseDists[j] = mix(theseDists[j], prevDist, edgeDecider);
            }
        #endif
    } else if (theseDists[j] > 0.0) {
        theseDists[j] = fullDist[localCoord.x+1][localCoord.y+1][localCoord.z+1] - distScale;
    }
    barrier();
}
