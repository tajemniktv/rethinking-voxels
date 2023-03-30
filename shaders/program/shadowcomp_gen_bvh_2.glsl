layout(local_size_x = 1) in;

const ivec3 workGroups = ivec3(65536, 1, 1);

#include "/lib/common.glsl"
#include "/lib/vx/SSBOs.glsl"

void main() {
	bvh_entry_t rawEntry;
	rawEntry.lower = vec3( 100000);
	rawEntry.upper = vec3(-100000);
	for (int k = 0; k < 4; k++) {
		rawEntry.children0[k] = 0;
		rawEntry.children1[k] = 0;
	}
	rawEntry.attachedTriLoc = 0;
	const int strideSize = max(1, MAX_TRIS / workGroups.x);
	int remaining = min(strideSize, numFaces - strideSize * int(gl_WorkGroupID.x));
	tri_t thisThreadTris[strideSize];
	vec3 avgPoss[strideSize];
	float sizes[strideSize];
	int entryIndices[strideSize];
	bool recurseFurther[strideSize];
	for (int i0 = 0; i0 < remaining; i0++) {
		int i = strideSize * int(gl_WorkGroupID.x) + i0;
		tri_t thisTri = tris[i];
		thisThreadTris[i0] = thisTri;
		vec3 lowerBound = min(min(
			thisTri.pos[0],
			thisTri.pos[1]),
			thisTri.pos[2]
		);
		vec3 upperBound = max(max(
			thisTri.pos[0],
			thisTri.pos[1]),
			thisTri.pos[2]
		);
		vec3 cnormal = cross(thisTri.pos[0] - thisTri.pos[1], thisTri.pos[0] - thisTri.pos[2]);
		avgPoss[i0] = 0.5 * (lowerBound + upperBound) - 0.01 * cnormal;
		sizes[i0] = dot(upperBound - lowerBound, vec3(0.33));
		entryIndices[i0] = 0;
		recurseFurther[i0] = true;
	}
	for (int depth = 0; depth < BVH_MAX_DEPTH; depth++) {
		int childNums[strideSize];
		bool newRecurseFurther[strideSize];
		for (int i0 = 0; i0 < remaining; i0++) {
			if (!recurseFurther[i0]) continue;
			bvh_entry_t thisEntry = bvhEntries[entryIndices[i0]];
			float entrySize = dot(thisEntry.upper - thisEntry.lower, vec3(0.33));
			newRecurseFurther[i0] = (entrySize > sizes[i0]);
			if (!newRecurseFurther[i0]) continue;
			vec3 localTriPos = (avgPoss[i0] - thisEntry.lower) / (thisEntry.upper - thisEntry.lower);
			ivec3 childNum0 = ivec3(greaterThan(localTriPos, vec3(0.5)));
			childNums[i0] = childNum0.x + (childNum0.y << 1) + (childNum0.z << 2);
			int childPointer = childNums[i0] >= 4 ? 
				atomicExchange(bvhEntries[entryIndices[i0]].children1[childNums[i0] - 4], -1) :
				atomicExchange(bvhEntries[entryIndices[i0]].children0[childNums[i0]], -1);
			if (childPointer == 0) {
				int newEntryIndex = atomicAdd(numBvhEntries, 1);
				childNums[i0] >= 4 ? 
					atomicExchange(bvhEntries[entryIndices[i0]].children1[childNums[i0] - 4], newEntryIndex) :
					atomicExchange(bvhEntries[entryIndices[i0]].children0[childNums[i0]], newEntryIndex);
				bvh_entry_t newEntry;
				newEntry.lower = mix(thisEntry.lower, thisEntry.upper, 0.5 * childNum0);
				newEntry.upper = mix(thisEntry.lower, thisEntry.upper, 0.5 * childNum0 + 0.5);
				newEntry.attachedTriLoc = 0;
				for (int k = 0; k < 4; k++) {
					newEntry.children0[k] = 0;
					newEntry.children1[k] = 0;
				}
				bvhEntries[newEntryIndex] = newEntry;
			} else if (childPointer > 0) {
				childNums[i0] >= 4 ? 
					atomicExchange(bvhEntries[entryIndices[i0]].children1[childNums[i0] - 4], childPointer) :
					atomicExchange(bvhEntries[entryIndices[i0]].children0[childNums[i0]], childPointer);
			}
		}
		memoryBarrierBuffer();
		for (int i0 = 0; i0 < remaining; i0++) {
			if (recurseFurther[i0]) {
				if (!newRecurseFurther[i0]) {
					recurseFurther[i0] = false;
				} else {
					entryIndices[i0] = getBvhChild(bvhEntries[entryIndices[i0]], childNums[i0]);
				}
			}
		}
	}
	for (int i0 = 0; i0 < remaining; i0++) {
		atomicAdd(bvhEntries[entryIndices[i0]].attachedTriLoc, 1);
		int i = strideSize * int(gl_WorkGroupID.x) + i0;
		tris[i].bvhParent = entryIndices[i0];
	}
	memoryBarrierBuffer();
	if (gl_WorkGroupID.x == 0) {
		int totalCount = 0;
		for (int i = 0; i < numBvhEntries; i++) {
			bvhLeaves[totalCount] = 1;
			totalCount += atomicExchange(bvhEntries[i].attachedTriLoc, totalCount) + 1;
		}
	}
	memoryBarrierBuffer();
	for (int i0 = 0; i0 < remaining; i0++) {
		int i = strideSize * int(gl_WorkGroupID.x) + i0;
		tri_t thisTri = tris[i];
		int globalLeafLoc = bvhEntries[thisTri.bvhParent].attachedTriLoc;
		int localLeafLoc = atomicAdd(bvhLeaves[globalLeafLoc], 1);
		int leafLoc = globalLeafLoc + localLeafLoc;
		if (thisTri.bvhParent == numBvhEntries - 1 || leafLoc < bvhEntries[thisTri.bvhParent + 1].attachedTriLoc) bvhLeaves[leafLoc] = i;
	}
}