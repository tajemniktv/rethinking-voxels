
	int leafCount = bvhLeaves[entryStack[depth].attachedTriLoc];
	int triLocEnd = entryStack[depth].attachedTriLoc + leafCount;
	for (int k = entryStack[depth].attachedTriLoc + 1; k <= triLocEnd; k++) {
		int thisTriId = bvhLeaves[k];
		tri_t thisTri = tris[thisTriId];
		if (backFaceCulling) {
			vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
			if (dot(cnormal, dir) >= 0) continue;
		}
		vec2 boundWs = boundsIntersect(POINTER_VOLUME_RES * pos0, POINTER_VOLUME_RES * dir, thisTri);
		if (boundWs.y <= 0 || boundWs.x > hitW) continue;
		vec3 hitPos = rayTriangleIntersect(POINTER_VOLUME_RES * pos0, POINTER_VOLUME_RES * dir, thisTri);
		if (hitPos.z <= 0 || hitPos.z > hitW) continue;
		ivec2 coord0 = ivec2(thisTri.texCoord[0] % 65536, thisTri.texCoord[0] / 65536);
		vec2 coord = coord0;
		vec2 offsetDir = vec2(0);
		for (int i = 0; i < 2; i++) {
			ivec2 coord1 = ivec2(thisTri.texCoord[i+1] % 65536, thisTri.texCoord[i+1] / 65536);
			vec2 dcoord = coord1 - coord0;
			dcoord += sign(dcoord);
			coord += vec2(hitPos[i] * dcoord);
			offsetDir += sign(dcoord) * (1 - abs(offsetDir));
		}
		coord -= 0.5 * offsetDir;
		vec4 newColor = ((thisTri.matBools >> 16) % 2 == 0) ? texelFetch(atlas, ivec2(coord + 0.5), 0) : vec4(1);
		if (newColor.a < 0.1) continue;
		vec4 vertexCol0 = vec4(
				thisTri.vertexCol[0] % 256,
			(thisTri.vertexCol[0] >>  8) % 256,
			(thisTri.vertexCol[0] >> 16) % 256,
			(thisTri.vertexCol[0] >> 24) % 256
		);
		vec4 vertexCol = vertexCol0;
		for (int i = 0; i < 2; i++) {
			vec4 vertexCol1 = vec4(
					thisTri.vertexCol[i+1] % 256,
				(thisTri.vertexCol[i+1] >>  8) % 256,
				(thisTri.vertexCol[i+1] >> 16) % 256,
				(thisTri.vertexCol[i+1] >> 24) % 256
			);
			vertexCol += hitPos[i] * (vertexCol1 - vertexCol0);
		}
		newColor *= vertexCol / 255.0;
		if (transHitW < hitPos.z) rayColor += (1 - rayColor.a) * newColor * vec2(1, newColor.a).yyyx;
		else           rayColor = newColor +  (1 - newColor.a) * rayColor * vec2(1, rayColor.a).yyyx;
		if (newColor.a > 0.9) {
			if (hitPos.z < hitW) {
				hitW = hitPos.z;
				returnVal.pos = pos0 + hitPos.z * dir;
				returnVal.triId = thisTriId;
			}
		} else if (hitPos.z < transHitW) {
			transHitW = hitPos.z;
			returnVal.transPos = pos0 + hitPos.z * dir;
			returnVal.transTriId = thisTriId;
			returnVal.transColor = newColor;
		}
	}
	depth++;

	for (int k = 0; k < childCount; k++) {
		ivec3 octreePos = ivec3(k & 1, (k & 2) >> 1, (k & 4) >> 2);
		octreePos = dirSgn * octreePos + (1 - dirSgn) * (1 - octreePos);
		int childIndex = octreePos.x + (octreePos.y << 1) + (octreePos.z << 2);
		if (getBvhChild(entryStack[depth-1], childIndex) == 0) continue;
		entryStack[depth] = bvhEntries[getBvhChild(entryStack[depth-1], childIndex)];
		vec2 entryRayIsct = boundsIntersect(pos0, dir, entryStack[depth-1].lower, entryStack[depth-1].upper);
		if (entryRayIsct.x > hitW || entryRayIsct.y < 0) continue;
	}