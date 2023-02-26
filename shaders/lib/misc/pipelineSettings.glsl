/*
const int colortex0Format = R11F_G11F_B10F;	//main color
const int colortex1Format = RGB8;			//smoothnessD & materialMask & skyLightFactor
const int colortex2Format = RGBA16;			//taa, previous depth
const int colortex3Format = RGB8;			//*cloud texture on deferred* & translucentMult & bloom & final color
const int colortex4Format = RGBA8;			//volumetric cloud linear depth & volumetric light factor & normals in composite
const int colortex5Format = RGBA8_SNORM;	//normalM & scene image for water reflections
const int colortex6Format = R8;				//*cloud texture on gbuffers*
#ifdef TEMPORAL_FILTER
const int colortex7Format = RGBA16F;		//temporal filter
#endif
// voxel data
const int shadowcolor0Format = RGBA16;
const int shadowcolor1Format = RGBA16;
const int colortex8Format = RGBA16;
const int colortex9Format = RGBA16;
const int colortex10Format = RGBA16;
const int colortex11Format = RGBA16;
const int colortex12Format = RGBA16;		//previous frame lighting
const int colortex13Format = RGBA16;
*/

const bool colortex0Clear = true;
const bool colortex1Clear = true;
const bool colortex2Clear = false;
const bool colortex3Clear = true;
const bool colortex4Clear = false;
const bool colortex5Clear = false;
const bool colortex6Clear = false;
#ifdef TEMPORAL_FILTER
const bool colortex7Clear = false;
#endif
// temporal voxel data such as flood fill
const bool colortex8Clear = false;
const bool colortex9Clear = false;
const bool colortex10Clear = false;
const bool colortex11Clear = false;
const bool colortex12Clear = false;
const bool colortex13Clear = false;

const int noiseTextureResolution = 128;

const bool shadowHardwareFiltering = true;
const float shadowDistanceRenderMul = 1.0;
const float entityShadowDistanceMul = 1.0; // Iris devs may bless us with their power

const float drynessHalflife = 300.0;
const float wetnessHalflife = 300.0;

const float ambientOcclusionLevel = 1.0;