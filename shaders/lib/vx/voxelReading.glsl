uniform sampler3D distanceField;
layout(r32i) uniform restrict iimage3D occupancyVolume;
layout(r32i) uniform restrict iimage3D voxelCols;

#include "/lib/util/random.glsl"

int getVoxelResolution(vec3 pos) {
    float dist = infnorm(pos/(voxelVolumeSize-2.01));
    return clamp(int(-log2(dist))-1, 0, VOXEL_DETAIL_AMOUNT-1);
}

float getDistanceField(vec3 pos) {
    int resolution = getVoxelResolution(pos);
    pos = clamp((1<<resolution) * pos / voxelVolumeSize + 0.5, 0.5/voxelVolumeSize, 1-0.5/voxelVolumeSize);
    pos.y = 0.25 * (pos.y + (frameCounter+1)%2 * 2 + resolution/4);
    return texture(distanceField, pos)[resolution%4];
}

vec3 distanceFieldGradient(vec3 pos) {
    const float epsilon = 0.5/(1<<VOXEL_DETAIL_AMOUNT);
    // Tetrahedron technique: 4 samples instead of 6
    const vec3 k0 = vec3(1.0, -1.0, -1.0);
    const vec3 k1 = vec3(-1.0, -1.0, 1.0);
    const vec3 k2 = vec3(-1.0, 1.0, -1.0);
    const vec3 k3 = vec3(1.0, 1.0, 1.0);

    return normalize(
        k0 * getDistanceField(pos + k0 * epsilon) +
        k1 * getDistanceField(pos + k1 * epsilon) +
        k2 * getDistanceField(pos + k2 * epsilon) +
        k3 * getDistanceField(pos + k3 * epsilon)
    );
}

vec4 getColor(vec3 pos) {
    ivec3 coords = ivec3(pos + 0.5 * voxelVolumeSize);
    if (any(lessThan(coords, ivec3(0))) || any(greaterThanEqual(coords, voxelVolumeSize))) {
        return vec4(0);
    }
    // Compute scaled coords once
    ivec3 scaledCoords = coords * ivec3(1, 2, 1);
    ivec2 rawCol = ivec2(
        imageLoad(voxelCols, scaledCoords).r,
        imageLoad(voxelCols, scaledCoords + ivec3(0, 1, 0)).r
    );

    // Use bitwise AND instead of modulo for power-of-2
    const int mask13 = 0x1FFF; // (1<<13) - 1
    const int mask10 = 0x3FF;  // (1<<10) - 1

    vec4 col = vec4(
        float(rawCol.r & mask13),
        float((rawCol.r >> 13) & mask13),
        float(rawCol.g & mask13),
        float((rawCol.g >> 13) & mask10)
    );

    float divisor = float(rawCol.g >> 23);
    float maxCol = max(max(col.r, col.g), col.b);
    col.rgb /= max(20.0 * divisor, maxCol);
    col.a /= max(4.0 * divisor, maxCol * 0.2);
    col.a = 1.0 - col.a;
    return col;
}

int getLightLevel(ivec3 coords) {
    return imageLoad(occupancyVolume, coords).r >> 6 & 15; //FIXME not implemented
}

vec3 rayTrace(vec3 start, vec3 dir, float dither) {
    float dirLen = infnorm(dir);
    dir /= dirLen;
    vec3 startOffset = 0.001 * dir;
    float w = 0.001 + dither * getDistanceField(start + startOffset);

    for (int k = 0; k < RT_STEPS; k++) {
        if (w >= dirLen) break;

        float thisdist = getDistanceField(start + w * dir);
        if (abs(thisdist) < 0.0001) break;

        w += thisdist;
    }
    return start + min(w, dirLen) * dir;
}
vec4 coneTrace(vec3 start, vec3 dir, float angle, float dither) {
    const float angle0 = angle;
    const float angleThreshold = 0.01 * angle0;
    float dirLen = infnorm(dir);
    dir /= dirLen;
    vec3 startOffset = 0.001 * dir;
    float w = 0.001 + dither * getDistanceField(start + startOffset);
    vec4 color = vec4(0.0);

    #ifdef TRANSLUCENT_LIGHT_TINT
    const ivec3 voxelOffset = voxelVolumeSize/2 + 1000;
    #endif

    for (int k = 0; k < RT_STEPS; k++) {
        if (angle < angleThreshold || w > dirLen) break;

        vec3 thisPos = start + w * dir;
        float thisdist = getDistanceField(thisPos);

        #ifdef DIRECTION_UPDATING_CONETRACE
            float angleW = angle * w;
            if (thisdist < angleW) {
                vec3 dfGrad = distanceFieldGradient(thisPos);
                dfGrad = normalize(dfGrad - dot(dir, dfGrad) * dir);
                if (!any(isnan(dfGrad))) {
                    float offsetLen = 0.5 * max(0.0, angleW - thisdist);
                    dir = normalize(dir + (offsetLen/w) * dfGrad);
                    thisPos = start + w * dir;
                    thisdist += offsetLen;
                }
                angle = min(angle, thisdist / w);
            }
        #else
            angle = min(angle, thisdist / w);
        #endif

        #ifdef TRANSLUCENT_LIGHT_TINT
        if (thisdist < 0.75) {
            ivec3 coords = ivec3(thisPos) + voxelOffset;
            if ((imageLoad(occupancyVolume, coords).r & 256) != 0) {
                vec4 localCol = getColor(thisPos);
                float weight = max(0.0, 1.2 * min(2.0 * localCol.a, 2.0 - 2.0 * localCol.a) - 0.2);
                color.rgb += localCol.rgb * weight;
                color.a += weight;
            }
        }
        #endif
        w += thisdist;
    }

    float dirLenThreshold = dirLen * 0.97;
    return vec4(
        angle > angleThreshold ?
        mix(vec3(1.0), color.rgb / max(color.a, 0.0001), min(1.0, color.a * 2.0)) :
        start + min(w, dirLen) * dir,
        max(0.0, float(w > dirLenThreshold) * (angle/angle0 - 0.01) / 0.99));
}

vec4 voxelTrace(vec3 start, vec3 dir, out vec3 normal, int hitMask) {
    dir += 0.000001 * vec3(equal(dir, vec3(0)));
    const vec3 stp = 1.0 / abs(dir);
    const vec3 dirsgn = sign(dir);
    const ivec3 voxelCenterOffset = voxelVolumeSize/2;
    vec3 progress = (0.5 + 0.5 * dirsgn - fract(start)) * stp * dirsgn;
    float w = 0.000001;
    normal = vec3(0);

    for (int k = 0; k < 2000; k++) {
        vec3 thisVoxelPos = start + w * dir;
        ivec3 thisVoxelCoord = ivec3(thisVoxelPos + 0.5 * normal * dirsgn) + voxelCenterOffset;

        // Combined bounds check
        if (any(greaterThanEqual(thisVoxelCoord, voxelVolumeSize)) || any(lessThan(thisVoxelCoord, ivec3(0)))) {
            break;
        }

        int thisVoxelData = imageLoad(occupancyVolume, thisVoxelCoord).r;
        int maskedData = thisVoxelData & hitMask;
        if (w > 1.0 || maskedData != 0) {
            normal *= -dirsgn;
            return vec4(thisVoxelPos, float(maskedData));
        }

        w = min(progress.x, min(progress.y, progress.z));
        normal = vec3(equal(progress, vec3(w)));
        progress += normal * stp;
    }
    return vec4(-10000.0);
}

