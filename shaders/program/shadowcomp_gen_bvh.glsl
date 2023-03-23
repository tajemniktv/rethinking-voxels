layout(local_size_x = 1) in;

const ivec3 workGroups = ivec3(65536, 1, 1);

#include "/lib/common.glsl"
#include "/lib/vx/SSBOs.glsl"

void main() {
	bvh_entry_t rawEntry;
	rawEntry.lower = vec3( 100000);
	rawEntry.upper = vec3(-100000);
	for (int k = 0; k < 8; k++) rawEntry.children[k] = 0;
	rawEntry.attachedTriLoc = 0;
	int remaining = min((MAX_TRIS / workGroups.x), int(numFaces - (MAX_TRIS / workGroups.x) * gl_WorkGroupID.x));
	for (int i0 = 0; i0 < remaining; i0++) {
		int i = (MAX_TRIS / workGroups.x) * int(gl_WorkGroupID.x) + i0;
		tri_t thisTri = tris[i];
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
		vec3 avgPos = 0.5 * (lowerBound + upperBound);
		float size = 0.33 * dot(upperBound - lowerBound, vec3(1));
		int entryIndex = 0;
		bool abort = false;
		for (int k = 0; k < 16; k++) {
			bvh_entry_t thisEntry = bvhEntries[entryIndex];
			float entrySize = 0.33 * dot(thisEntry.upper - thisEntry.lower, vec3(1));
			bool recurseFurther = (entrySize > size);
			for (int i = 0; i < 1; i++) {
				if (!recurseFurther) break;
				vec3 localTriPos = (avgPos - thisEntry.lower) / (thisEntry.upper - thisEntry.lower);
				ivec3 childNum0 = ivec3(localTriPos.x > 0.5, localTriPos.y > 0.5, localTriPos.z > 0.5);
				int childNum = childNum0.x + (childNum0.y << 1) + (childNum0.z << 2);
				int childPointer = atomicExchange(bvhEntries[entryIndex].children[childNum], -1);
				if (childPointer == -1) {
					recurseFurther = false;
					break;
				}
				if (childPointer == 0) {
					bvh_entry_t newEntry = rawEntry;
					newEntry.lower = mix(thisEntry.lower, thisEntry.upper, 0.5 * childNum0);
					newEntry.upper = mix(thisEntry.lower, thisEntry.upper, 0.5 * childNum0 + 0.5);
					int newEntryIndex = atomicAdd(numBvhEntries, 1);
					bvhEntries[newEntryIndex] = newEntry;
					atomicExchange(bvhEntries[entryIndex].children[childNum], newEntryIndex);
					entryIndex = newEntryIndex;
					break;
				}
				entryIndex = childPointer;
			}
			if (!recurseFurther) {
				atomicAdd(bvhEntries[entryIndex].attachedTriLoc, 1);
				tris[i].bvhParent = entryIndex;
				break;
			}
		}
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
		int i = (MAX_TRIS / workGroups.x) * int(gl_WorkGroupID.x) + i0;
		tri_t thisTri = tris[i];
		int globalLeafLoc = bvhEntries[thisTri.bvhParent].attachedTriLoc;
		int localLeafLoc = atomicAdd(bvhLeaves[globalLeafLoc], 1);
		bvhLeaves[globalLeafLoc + localLeafLoc] = i;
	}
}