layout(local_size_x = 1) in;

const ivec3 workGroups = ivec3(65536, 1, 1);

#include "/lib/common.glsl"
#include "/lib/vx/SSBOs.glsl"

void main() {
	// clear enough space to store bin sizes in
	sortingStuff[gl_WorkGroupID.x][2] = 0;
	int faceCount = numFaces;
	memoryBarrierBuffer();
	// get info about this thread's tris, determine bin sizes
	int localIndex[MAX_TRIS / 65536];
	int globalIndex[MAX_TRIS / 65536];
	for (int k = 0; k < MAX_TRIS / 65536; k++) {
		int arrayIndex0 = (MAX_TRIS / 65536) * int(gl_WorkGroupID.x) + k;
		int arrayIndex1 = (tris[arrayIndex0].mortonCode >> 14);
		int binSize = atomicAdd(sortingStuff[arrayIndex1][2], 1);
		localIndex[k] = binSize;
		globalIndex[k] = arrayIndex1;
	}
	memoryBarrierBuffer();
	// get start and end indices of the bins
	if (gl_WorkGroupID.x == 0) {
		int currentIndex = 0;
		for (int k = 0; k < 65537; k++) currentIndex += atomicExchange(sortingStuff[k][2], currentIndex);
	}
	memoryBarrierBuffer();
	// write bin-sorted data into the array
	for (int k = 0; k < MAX_TRIS / 65536; k++) {
		sortingStuff[sortingStuff[globalIndex[k]][2] + localIndex[k]][0] = (MAX_TRIS / 65536) * int(gl_WorkGroupID.x) + k;
	}
	// sort the bins
	if (gl_WorkGroupID.x == 0) {
		sortingStuff[0][1] = 1;
		sortingStuff[1][1] = 1;
		sortingStuff[0][2] = 1;
		sortingStuff[1][2] = 1;
	}
	memoryBarrierBuffer();
	ivec2 thisBinBounds0 = ivec2(sortingStuff[gl_WorkGroupID.x][2], sortingStuff[gl_WorkGroupID.x + 1][2]);
	if (thisBinBounds0.y > thisBinBounds0.x + 1) {
		int openBinNum = atomicAdd(sortingStuff[0][1], 1);
		sortingStuff[2 * openBinNum    ][1] = thisBinBounds0.x;
		sortingStuff[2 * openBinNum + 1][1] = thisBinBounds0.y;
	}
	memoryBarrierBuffer();
	for (int i = 13; i >= 0; i--) {
		int relevantBit = (1<<i);
		int locBins = 2 - i % 2;
		int newLocBins = 3 - locBins;
		while (true) {
			int currentSortingBin = atomicAdd(sortingStuff[1][locBins], 1);
			if (currentSortingBin >= sortingStuff[0][locBins]) break;
			ivec2 thisBinBounds = ivec2(
				sortingStuff[2 * currentSortingBin    ][locBins],
				sortingStuff[2 * currentSortingBin + 1][locBins]
			);
			int thisBinSize = thisBinBounds.y - thisBinBounds.x;
			ivec2 itemCounts = ivec2(0, 1);
			int loc = thisBinBounds.x;
			int thisTriId = sortingStuff[loc][0];
			for (int k = 0; k < thisBinSize; k++) {
				int upper = int((tris[thisTriId].mortonCode & relevantBit) != 0);
				int newLoc = thisBinBounds[upper] + (1 - 2 * upper) * itemCounts[upper];
				itemCounts[upper]++;
				if (loc == newLoc) {
					loc += 1 - 2 * upper;
					thisTriId = sortingStuff[loc][0];
				} else {
					loc = newLoc;
					int newTriId = sortingStuff[loc][0];
					sortingStuff[loc][0] = thisTriId;
					thisTriId = newTriId;
				}
			}
			if (itemCounts.x > 1) {
				int newBinId = atomicAdd(sortingStuff[0][newLocBins], 1);
				sortingStuff[2 * newBinId    ][newLocBins] = thisBinBounds.x;
				sortingStuff[2 * newBinId + 1][newLocBins] = thisBinBounds.x + itemCounts.x;
			}
			if (itemCounts.y > 2) {
				int newBinId = atomicAdd(sortingStuff[0][newLocBins], 1);
				sortingStuff[2 * newBinId    ][newLocBins] = thisBinBounds.x + itemCounts.x;
				sortingStuff[2 * newBinId + 1][newLocBins] = thisBinBounds.y;
			}
		}
		memoryBarrierBuffer();
		if (gl_WorkGroupID.x == 0) {
			sortingStuff[0][locBins] = 1;
			sortingStuff[1][locBins] = 1;
		}
		memoryBarrierBuffer();
	}
	if (gl_WorkGroupID.x == 0) {
		sortingStuff[0][1] = 2;
		sortingStuff[1][1] = 1;
		sortingStuff[0][2] = 1;
		sortingStuff[1][2] = 1;
		sortingStuff[3][1] = 0;
		sortingStuff[4][1] = numFaces;
		sortingStuff[5][1] = -1;
	}
	memoryBarrierBuffer();
	// fill the BVH
	for (int k = 0; k < 15; k++) {
		int locBins = 1 + k % 2;
		int newLocBins = 3 - locBins;
		if (gl_WorkGroupID.x >= 1 << (3 * k)) {
			
		}
		while(true) {
			int i0 = atomicAdd(sortingStuff[1][locBins], 1);
			if (i0 >= sortingStuff[0][locBins]) break;
			int lower  = sortingStuff[3 * i0    ][locBins];
			int upper  = sortingStuff[3 * i0 + 1][locBins];
			int parent = sortingStuff[3 * i0 + 2][locBins];
			bvh_entry_t thisEntry;
			thisEntry.lower = vec3( 100000);
			thisEntry.upper = vec3(-100000);
			int BvhIndex = atomicAdd(numBvhEntries, 1);
			int diff = upper - lower;
			if (diff < 8) {
				thisEntry.childNum_isLeaf = 16 + diff;
				for (int i = 0; i < diff; i++) {
					thisEntry.children[i] = sortingStuff[lower + i][0];
				}
			} else {
				thisEntry.childNum_isLeaf = 0;
				int newLower = lower;
				for (int i = 1; i <= 8; i++) {
					int i1 = atomicAdd(sortingStuff[0][newLocBins], 1);
					sortingStuff[3 * i1    ][newLocBins] = newLower;
					newLower = lower + diff * i / 8;
					sortingStuff[3 * i1 + 1][newLocBins] = newLower;
					sortingStuff[3 * i1 + 2][newLocBins] = BvhIndex;
				}
			}
			if (parent > 0) {
				int numBorn = atomicAdd(bvhEntries[parent].childNum_isLeaf, 1);
				bvhEntries[parent].children[numBorn] = BvhIndex;
			}
			bvhEntries[BvhIndex] = thisEntry;
		}
		memoryBarrierBuffer();
		if (gl_WorkGroupID.x == 0) {
			sortingStuff[0][locBins] = 1;
			sortingStuff[1][locBins] = 1;
		}
		memoryBarrierBuffer();
	}
}