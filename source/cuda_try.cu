#include <cuComplex.h>
#include <stdio.h>
#ifndef check_cuda_errors
#define check_cuda_errors(err) __check_cuda_errors(err, __FILE__, __LINE__)
void __check_cuda_errors(cudaError_t err, const char *file, const int line) {
  if ((cudaSuccess != err)) {
    fprintf(stderr,
            "cuda driver api errror: %04d '%s' from file <%s>, line %i.\n", err,
            cudaGetErrorString(err), file, line);
  }
}
#endif
// https://www.youtube.com/watch?v=Ed_h2km0liI CUDACast #2 - Your First CUDA C
// Program
// https://github.com/NVIDIA-developer-blog/cudacasts/blob/master/ep2-first-cuda-c-program/kernel.cu
__global__ void vector_add(int *a, int *b, int *c, int n) {
  {
    int i = threadIdx.x;
    if ((i < n)) {
      c[i] = (a[i] + b[i]);
    }
  }
}
enum { N = 1024, NX = 256, NY = 32 };

__global__ void fft(cuFloatComplex *__restrict__ in) {
  {
    __shared__ cuFloatComplex tmp[NX];
    tmp[0] = in[0];
    tmp[128] = in[1];
    tmp[64] = in[2];
    tmp[192] = in[3];
    tmp[32] = in[4];
    tmp[160] = in[5];
    tmp[96] = in[6];
    tmp[224] = in[7];
    tmp[16] = in[8];
    tmp[144] = in[9];
    tmp[80] = in[10];
    tmp[208] = in[11];
    tmp[48] = in[12];
    tmp[176] = in[13];
    tmp[112] = in[14];
    tmp[240] = in[15];
    tmp[8] = in[16];
    tmp[136] = in[17];
    tmp[72] = in[18];
    tmp[200] = in[19];
    tmp[40] = in[20];
    tmp[168] = in[21];
    tmp[104] = in[22];
    tmp[232] = in[23];
    tmp[24] = in[24];
    tmp[152] = in[25];
    tmp[88] = in[26];
    tmp[216] = in[27];
    tmp[56] = in[28];
    tmp[184] = in[29];
    tmp[120] = in[30];
    tmp[248] = in[31];
    tmp[4] = in[32];
    tmp[132] = in[33];
    tmp[68] = in[34];
    tmp[196] = in[35];
    tmp[36] = in[36];
    tmp[164] = in[37];
    tmp[100] = in[38];
    tmp[228] = in[39];
    tmp[20] = in[40];
    tmp[148] = in[41];
    tmp[84] = in[42];
    tmp[212] = in[43];
    tmp[52] = in[44];
    tmp[180] = in[45];
    tmp[116] = in[46];
    tmp[244] = in[47];
    tmp[12] = in[48];
    tmp[140] = in[49];
    tmp[76] = in[50];
    tmp[204] = in[51];
    tmp[44] = in[52];
    tmp[172] = in[53];
    tmp[108] = in[54];
    tmp[236] = in[55];
    tmp[28] = in[56];
    tmp[156] = in[57];
    tmp[92] = in[58];
    tmp[220] = in[59];
    tmp[60] = in[60];
    tmp[188] = in[61];
    tmp[124] = in[62];
    tmp[252] = in[63];
    tmp[2] = in[64];
    tmp[130] = in[65];
    tmp[66] = in[66];
    tmp[194] = in[67];
    tmp[34] = in[68];
    tmp[162] = in[69];
    tmp[98] = in[70];
    tmp[226] = in[71];
    tmp[18] = in[72];
    tmp[146] = in[73];
    tmp[82] = in[74];
    tmp[210] = in[75];
    tmp[50] = in[76];
    tmp[178] = in[77];
    tmp[114] = in[78];
    tmp[242] = in[79];
    tmp[10] = in[80];
    tmp[138] = in[81];
    tmp[74] = in[82];
    tmp[202] = in[83];
    tmp[42] = in[84];
    tmp[170] = in[85];
    tmp[106] = in[86];
    tmp[234] = in[87];
    tmp[26] = in[88];
    tmp[154] = in[89];
    tmp[90] = in[90];
    tmp[218] = in[91];
    tmp[58] = in[92];
    tmp[186] = in[93];
    tmp[122] = in[94];
    tmp[250] = in[95];
    tmp[6] = in[96];
    tmp[134] = in[97];
    tmp[70] = in[98];
    tmp[198] = in[99];
    tmp[38] = in[100];
    tmp[166] = in[101];
    tmp[102] = in[102];
    tmp[230] = in[103];
    tmp[22] = in[104];
    tmp[150] = in[105];
    tmp[86] = in[106];
    tmp[214] = in[107];
    tmp[54] = in[108];
    tmp[182] = in[109];
    tmp[118] = in[110];
    tmp[246] = in[111];
    tmp[14] = in[112];
    tmp[142] = in[113];
    tmp[78] = in[114];
    tmp[206] = in[115];
    tmp[46] = in[116];
    tmp[174] = in[117];
    tmp[110] = in[118];
    tmp[238] = in[119];
    tmp[30] = in[120];
    tmp[158] = in[121];
    tmp[94] = in[122];
    tmp[222] = in[123];
    tmp[62] = in[124];
    tmp[190] = in[125];
    tmp[126] = in[126];
    tmp[254] = in[127];
    tmp[1] = in[128];
    tmp[129] = in[129];
    tmp[65] = in[130];
    tmp[193] = in[131];
    tmp[33] = in[132];
    tmp[161] = in[133];
    tmp[97] = in[134];
    tmp[225] = in[135];
    tmp[17] = in[136];
    tmp[145] = in[137];
    tmp[81] = in[138];
    tmp[209] = in[139];
    tmp[49] = in[140];
    tmp[177] = in[141];
    tmp[113] = in[142];
    tmp[241] = in[143];
    tmp[9] = in[144];
    tmp[137] = in[145];
    tmp[73] = in[146];
    tmp[201] = in[147];
    tmp[41] = in[148];
    tmp[169] = in[149];
    tmp[105] = in[150];
    tmp[233] = in[151];
    tmp[25] = in[152];
    tmp[153] = in[153];
    tmp[89] = in[154];
    tmp[217] = in[155];
    tmp[57] = in[156];
    tmp[185] = in[157];
    tmp[121] = in[158];
    tmp[249] = in[159];
    tmp[5] = in[160];
    tmp[133] = in[161];
    tmp[69] = in[162];
    tmp[197] = in[163];
    tmp[37] = in[164];
    tmp[165] = in[165];
    tmp[101] = in[166];
    tmp[229] = in[167];
    tmp[21] = in[168];
    tmp[149] = in[169];
    tmp[85] = in[170];
    tmp[213] = in[171];
    tmp[53] = in[172];
    tmp[181] = in[173];
    tmp[117] = in[174];
    tmp[245] = in[175];
    tmp[13] = in[176];
    tmp[141] = in[177];
    tmp[77] = in[178];
    tmp[205] = in[179];
    tmp[45] = in[180];
    tmp[173] = in[181];
    tmp[109] = in[182];
    tmp[237] = in[183];
    tmp[29] = in[184];
    tmp[157] = in[185];
    tmp[93] = in[186];
    tmp[221] = in[187];
    tmp[61] = in[188];
    tmp[189] = in[189];
    tmp[125] = in[190];
    tmp[253] = in[191];
    tmp[3] = in[192];
    tmp[131] = in[193];
    tmp[67] = in[194];
    tmp[195] = in[195];
    tmp[35] = in[196];
    tmp[163] = in[197];
    tmp[99] = in[198];
    tmp[227] = in[199];
    tmp[19] = in[200];
    tmp[147] = in[201];
    tmp[83] = in[202];
    tmp[211] = in[203];
    tmp[51] = in[204];
    tmp[179] = in[205];
    tmp[115] = in[206];
    tmp[243] = in[207];
    tmp[11] = in[208];
    tmp[139] = in[209];
    tmp[75] = in[210];
    tmp[203] = in[211];
    tmp[43] = in[212];
    tmp[171] = in[213];
    tmp[107] = in[214];
    tmp[235] = in[215];
    tmp[27] = in[216];
    tmp[155] = in[217];
    tmp[91] = in[218];
    tmp[219] = in[219];
    tmp[59] = in[220];
    tmp[187] = in[221];
    tmp[123] = in[222];
    tmp[251] = in[223];
    tmp[7] = in[224];
    tmp[135] = in[225];
    tmp[71] = in[226];
    tmp[199] = in[227];
    tmp[39] = in[228];
    tmp[167] = in[229];
    tmp[103] = in[230];
    tmp[231] = in[231];
    tmp[23] = in[232];
    tmp[151] = in[233];
    tmp[87] = in[234];
    tmp[215] = in[235];
    tmp[55] = in[236];
    tmp[183] = in[237];
    tmp[119] = in[238];
    tmp[247] = in[239];
    tmp[15] = in[240];
    tmp[143] = in[241];
    tmp[79] = in[242];
    tmp[207] = in[243];
    tmp[47] = in[244];
    tmp[175] = in[245];
    tmp[111] = in[246];
    tmp[239] = in[247];
    tmp[31] = in[248];
    tmp[159] = in[249];
    tmp[95] = in[250];
    tmp[223] = in[251];
    tmp[63] = in[252];
    tmp[191] = in[253];
    tmp[127] = in[254];
    tmp[255] = in[255];
  }
}
void cuda_list_attributes(int cuda_dev) {
  {
    int val;
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxThreadsPerBlock, cuda_dev));
    printf("cudaDevAttrMaxThreadsPerBlock.................... = %12d (Maximum "
           "number of threads per block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimX, cuda_dev));
    printf("cudaDevAttrMaxBlockDimX.......................... = %12d (Maximum "
           "x-dimension of a block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimY, cuda_dev));
    printf("cudaDevAttrMaxBlockDimY.......................... = %12d (Maximum "
           "y-dimension of a block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimZ, cuda_dev));
    printf("cudaDevAttrMaxBlockDimZ.......................... = %12d (Maximum "
           "z-dimension of a block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimX, cuda_dev));
    printf("cudaDevAttrMaxGridDimX........................... = %12d (Maximum "
           "x-dimension of a grid)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimY, cuda_dev));
    printf("cudaDevAttrMaxGridDimY........................... = %12d (Maximum "
           "y-dimension of a grid)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimZ, cuda_dev));
    printf("cudaDevAttrMaxGridDimZ........................... = %12d (Maximum "
           "z-dimension of a grid)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSharedMemoryPerBlock, cuda_dev));
    printf("cudaDevAttrMaxSharedMemoryPerBlock............... = %12d (Maximum "
           "amount of shared memoryavailable to a thread block in bytes)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrTotalConstantMemory, cuda_dev));
    printf("cudaDevAttrTotalConstantMemory................... = %12d (Memory "
           "available on device for __constant__variables in a CUDA C kernel "
           "in bytes)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrWarpSize, cuda_dev));
    printf("cudaDevAttrWarpSize.............................. = %12d (Warp "
           "size in threads)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxPitch, cuda_dev));
    printf("cudaDevAttrMaxPitch.............................. = %12d (Maximum "
           "pitch in bytes allowed by the memory copyfunctions that involve "
           "memory regions allocated through cudaMallocPitch())\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture1DWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DWidth..................... = %12d (Maximum "
           "1D texture width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DLinearWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DLinearWidth............... = %12d (Maximum "
           "width for a 1D texture boundto linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DMipmappedWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DMipmappedWidth............ = %12d (Maximum "
           "mipmapped 1D texturewidth)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DWidth..................... = %12d (Maximum "
           "2D texture width)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DHeight.................... = %12d (Maximum "
           "2D texture height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLinearWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLinearWidth............... = %12d (Maximum "
           "width for a 2D texture boundto linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLinearHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLinearHeight.............. = %12d (Maximum "
           "height for a 2D texture boundto linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLinearPitch, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLinearPitch............... = %12d (Maximum "
           "pitch in bytes for a 2D texturebound to linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DMipmappedWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DMipmappedWidth............ = %12d (Maximum "
           "mipmapped 2D texturewidth)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DMipmappedHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DMipmappedHeight........... = %12d (Maximum "
           "mipmapped 2D textureheight)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture3DWidth..................... = %12d (Maximum "
           "3D texture width)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture3DHeight.................... = %12d (Maximum "
           "3D texture height)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DDepth, cuda_dev));
    printf("cudaDevAttrMaxTexture3DDepth..................... = %12d (Maximum "
           "3D texture depth)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture3DWidthAlt, cuda_dev));
    printf("cudaDevAttrMaxTexture3DWidthAlt.................. = %12d "
           "(Alternate maximum 3D texture width, 0 if noalternate maximum 3D "
           "texture size is supported)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture3DHeightAlt, cuda_dev));
    printf("cudaDevAttrMaxTexture3DHeightAlt................. = %12d "
           "(Alternate maximum 3D texture height, 0 ifno alternate maximum 3D "
           "texture size is supported)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture3DDepthAlt, cuda_dev));
    printf("cudaDevAttrMaxTexture3DDepthAlt.................. = %12d "
           "(Alternate maximum 3D texture depth, 0 if noalternate maximum 3D "
           "texture size is supported)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTextureCubemapWidth, cuda_dev));
    printf("cudaDevAttrMaxTextureCubemapWidth................ = %12d (Maximum "
           "cubemap texture width orheight)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DLayeredWidth.............. = %12d (Maximum "
           "1D layered texture width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxTexture1DLayeredLayers............. = %12d (Maximum "
           "layers in a 1D layeredtexture)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLayeredWidth.............. = %12d (Maximum "
           "2D layered texture width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLayeredHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLayeredHeight............. = %12d (Maximum "
           "2D layered texture height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLayeredLayers............. = %12d (Maximum "
           "layers in a 2D layeredtexture)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTextureCubemapLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxTextureCubemapLayeredWidth......... = %12d (Maximum "
           "cubemap layeredtexture width or height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTextureCubemapLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxTextureCubemapLayeredLayers........ = %12d (Maximum "
           "layers in a cubemaplayered texture)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface1DWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface1DWidth..................... = %12d (Maximum "
           "1D surface width)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface2DWidth..................... = %12d (Maximum "
           "2D surface width)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DHeight, cuda_dev));
    printf("cudaDevAttrMaxSurface2DHeight.................... = %12d (Maximum "
           "2D surface height)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface3DWidth..................... = %12d (Maximum "
           "3D surface width)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DHeight, cuda_dev));
    printf("cudaDevAttrMaxSurface3DHeight.................... = %12d (Maximum "
           "3D surface height)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DDepth, cuda_dev));
    printf("cudaDevAttrMaxSurface3DDepth..................... = %12d (Maximum "
           "3D surface depth)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface1DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface1DLayeredWidth.............. = %12d (Maximum "
           "1D layered surface width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface1DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxSurface1DLayeredLayers............. = %12d (Maximum "
           "layers in a 1D layeredsurface)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface2DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface2DLayeredWidth.............. = %12d (Maximum "
           "2D layered surface width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface2DLayeredHeight, cuda_dev));
    printf("cudaDevAttrMaxSurface2DLayeredHeight............. = %12d (Maximum "
           "2D layered surface height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface2DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxSurface2DLayeredLayers............. = %12d (Maximum "
           "layers in a 2D layeredsurface)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurfaceCubemapWidth, cuda_dev));
    printf("cudaDevAttrMaxSurfaceCubemapWidth................ = %12d (Maximum "
           "cubemap surface width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurfaceCubemapLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxSurfaceCubemapLayeredWidth......... = %12d (Maximum "
           "cubemap layeredsurface width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurfaceCubemapLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxSurfaceCubemapLayeredLayers........ = %12d (Maximum "
           "layers in a cubemaplayered surface)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxRegistersPerBlock, cuda_dev));
    printf("cudaDevAttrMaxRegistersPerBlock.................. = %12d (Maximum "
           "number of 32-bit registers availableto a thread block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrClockRate, cuda_dev));
    printf("cudaDevAttrClockRate............................. = %12d (Peak "
           "clock frequency in kilohertz)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrTextureAlignment, cuda_dev));
    printf(
        "cudaDevAttrTextureAlignment...................... = %12d (Alignment "
        "requirement texture base addressesaligned to textureAlign bytes do "
        "not need an offset applied to texture fetches)\n",
        val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrTexturePitchAlignment, cuda_dev));
    printf("cudaDevAttrTexturePitchAlignment................. = %12d (Pitch "
           "alignment requirement for 2D texturereferences bound to pitched "
           "memory)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrGpuOverlap, cuda_dev));
    printf("cudaDevAttrGpuOverlap............................ = %12d (1 if the "
           "device can concurrently copy memory betweenhost and device while "
           "executing a kernel, or 0 if not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount, cuda_dev));
    printf("cudaDevAttrMultiProcessorCount................... = %12d (Number "
           "of multiprocessors on the device)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrKernelExecTimeout, cuda_dev));
    printf(
        "cudaDevAttrKernelExecTimeout..................... = %12d (1 if there "
        "is a run time limit for kernels executedon the device, or 0 if not)\n",
        val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, cuda_dev));
    printf("cudaDevAttrIntegrated............................ = %12d (1 if the "
           "device is integrated with the memory subsystem, or0 if not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrCanMapHostMemory, cuda_dev));
    printf(
        "cudaDevAttrCanMapHostMemory...................... = %12d (1 if the "
        "device can map host memory into theCUDA address space, or 0 if not)\n",
        val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrComputeMode, cuda_dev));
    printf("cudaDevAttrComputeMode........................... = %12d (Compute "
           "mode is the compute mode that the device iscurrently in. Available "
           "modes are as follows)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrConcurrentKernels, cuda_dev));
    printf("cudaDevAttrConcurrentKernels..................... = %12d (1 if the "
           "device supports executing multiple kernelswithin the same context "
           "simultaneously, or 0 if not. It is not guaranteed that "
           "multipkernels will be resident on the device concurrently so this "
           "feature should not berelied upon for correctness)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrEccEnabled, cuda_dev));
    printf("cudaDevAttrEccEnabled............................ = %12d (1 if "
           "error correction is enabled on the device, 0 if errorcorrection is "
           "disabled or not supported by the device)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrPciBusId, cuda_dev));
    printf("cudaDevAttrPciBusId.............................. = %12d (PCI bus "
           "identifier of the device)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrPciDeviceId, cuda_dev));
    printf("cudaDevAttrPciDeviceId........................... = %12d (PCI "
           "device (also known as slot) identifier of the device)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrTccDriver, cuda_dev));
    printf("cudaDevAttrTccDriver............................. = %12d (1 if the "
           "device is using a TCC driver. TCC is only availableon Tesla "
           "hardware running Windows Vista or later)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMemoryClockRate, cuda_dev));
    printf("cudaDevAttrMemoryClockRate....................... = %12d (Peak "
           "memory clock frequency in kilohertz)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrGlobalMemoryBusWidth, cuda_dev));
    printf("cudaDevAttrGlobalMemoryBusWidth.................. = %12d (Global "
           "memory bus width in bits)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrL2CacheSize, cuda_dev));
    printf("cudaDevAttrL2CacheSize........................... = %12d (Size of "
           "L2 cache in bytes. 0 if the device doesn't have L2cache)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxThreadsPerMultiProcessor, cuda_dev));
    printf("cudaDevAttrMaxThreadsPerMultiProcessor........... = %12d (Maximum "
           "resident threads permultiprocessor)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrUnifiedAddressing, cuda_dev));
    printf("cudaDevAttrUnifiedAddressing..................... = %12d (1 if the "
           "device shares a unified address space withthe host, or 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrComputeCapabilityMajor, cuda_dev));
    printf("cudaDevAttrComputeCapabilityMajor................ = %12d (Major "
           "compute capability version number)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrComputeCapabilityMinor, cuda_dev));
    printf("cudaDevAttrComputeCapabilityMinor................ = %12d (Minor "
           "compute capability version number)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrStreamPrioritiesSupported, cuda_dev));
    printf("cudaDevAttrStreamPrioritiesSupported............. = %12d (1 if the "
           "device supports stream priorities, or0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrGlobalL1CacheSupported, cuda_dev));
    printf("cudaDevAttrGlobalL1CacheSupported................ = %12d (1 if "
           "device supports caching globals in L1cache, 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrLocalL1CacheSupported, cuda_dev));
    printf("cudaDevAttrLocalL1CacheSupported................. = %12d (1 if "
           "device supports caching locals in L1cache, 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSharedMemoryPerMultiprocessor, cuda_dev));
    printf("cudaDevAttrMaxSharedMemoryPerMultiprocessor...... = %12d (Maximum "
           "amount of sharedmemory available to a multiprocessor in bytes this "
           "amount is shared by all threadblocks simultaneously resident on a "
           "multiprocessor)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxRegistersPerMultiprocessor, cuda_dev));
    printf("cudaDevAttrMaxRegistersPerMultiprocessor......... = %12d (Maximum "
           "number of 32-bitregisters available to a multiprocessor this "
           "number is shared by all thread blockssimultaneously resident on a "
           "multiprocessor)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrManagedMemory, cuda_dev));
    printf("cudaDevAttrManagedMemory......................... = %12d (1 if "
           "device supports allocating managed memory, 0 if not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrIsMultiGpuBoard, cuda_dev));
    printf("cudaDevAttrIsMultiGpuBoard....................... = %12d (1 if "
           "device is on a multi-GPU board, 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMultiGpuBoardGroupID, cuda_dev));
    printf("cudaDevAttrMultiGpuBoardGroupID.................. = %12d (Unique "
           "identifier for a group of devices onthe same multi-GPU board)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrHostNativeAtomicSupported, cuda_dev));
    printf("cudaDevAttrHostNativeAtomicSupported............. = %12d (1 if the "
           "link between the device and thehost supports native atomic "
           "operations)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrSingleToDoublePrecisionPerfRatio, cuda_dev));
    printf("cudaDevAttrSingleToDoublePrecisionPerfRatio...... = %12d (Ratio of "
           "single precisionperformance (in floating-point operations per "
           "second) to double precision performance)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrPageableMemoryAccess, cuda_dev));
    printf("cudaDevAttrPageableMemoryAccess.................. = %12d (1 if the "
           "device supports coherently accessingpageable memory without "
           "calling cudaHostRegister on it, and 0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrConcurrentManagedAccess, cuda_dev));
    printf("cudaDevAttrConcurrentManagedAccess............... = %12d (1 if the "
           "device can coherently accessmanaged memory concurrently with the "
           "CPU, and 0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrComputePreemptionSupported, cuda_dev));
    printf("cudaDevAttrComputePreemptionSupported............ = %12d (1 if the "
           "device supports ComputePreemption, 0 if not.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrCanUseHostPointerForRegisteredMem, cuda_dev));
    printf("cudaDevAttrCanUseHostPointerForRegisteredMem..... = %12d (1 if the "
           "device can access hostregistered memory at the same virtual "
           "address as the CPU, and 0 otherwise.)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrCooperativeLaunch, cuda_dev));
    printf("cudaDevAttrCooperativeLaunch..................... = %12d (1 if the "
           "device supports launching cooperativekernels via "
           "cudaLaunchCooperativeKernel, and 0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrCooperativeMultiDeviceLaunch, cuda_dev));
    printf("cudaDevAttrCooperativeMultiDeviceLaunch.......... = %12d (1 if the "
           "device supports launchingcooperative kernels via "
           "cudaLaunchCooperativeKernelMultiDevice, and 0otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrCanFlushRemoteWrites, cuda_dev));
    printf("cudaDevAttrCanFlushRemoteWrites.................. = %12d (1 if the "
           "device supports flushing ofoutstanding remote writes, and 0 "
           "otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrHostRegisterSupported, cuda_dev));
    printf("cudaDevAttrHostRegisterSupported................. = %12d (1 if the "
           "device supports host memoryregistration via cudaHostRegister, and "
           "0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrPageableMemoryAccessUsesHostPageTables, cuda_dev));
    printf("cudaDevAttrPageableMemoryAccessUsesHostPageTables = %12d (1 if the "
           "device accessespageable memory via the host's page tables, and 0 "
           "otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrDirectManagedMemAccessFromHost, cuda_dev));
    printf("cudaDevAttrDirectManagedMemAccessFromHost........ = %12d (1 if the "
           "host can directly accessmanaged memory on the device without "
           "migration, and 0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda_dev));
    printf("cudaDevAttrMaxSharedMemoryPerBlockOptin.......... = %12d (Maximum "
           "per blockshared memory size on the device. This value can be opted "
           "into when usingcudaFuncSetAttribute)\n",
           val);
  }
}
void cuda_list_limits(int cuda_dev) {
  {
    size_t val;
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitStackSize));
    printf("cudaLimitStackSize................... = %12lu (stack size in bytes "
           "of each GPU thread)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitPrintfFifoSize));
    printf("cudaLimitPrintfFifoSize.............. = %12lu (size in bytes of "
           "the shared FIFO used by the printf() devicesystem call)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitMallocHeapSize));
    printf("cudaLimitMallocHeapSize.............. = %12lu (size in bytes of "
           "the heap used by the malloc() and free()device system calls)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitDevRuntimeSyncDepth));
    printf(
        "cudaLimitDevRuntimeSyncDepth......... = %12lu (maximum grid depth at "
        "which a thread canisssue the device runtime call "
        "cudaDeviceSynchronize() to wait on child gridlaunches to complete.)\n",
        val);
    check_cuda_errors(
        cudaDeviceGetLimit(&val, cudaLimitDevRuntimePendingLaunchCount));
    printf("cudaLimitDevRuntimePendingLaunchCount = %12lu (maximum number of "
           "outstandingdevice runtime launches.)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitMaxL2FetchGranularity));
    printf("cudaLimitMaxL2FetchGranularity....... = %12lu (L2 cache fetch "
           "granularity.)\n",
           val);
  }
}
void cuda_list_properties(int cuda_dev) {
  {
    cudaDeviceProp device_prop;
    check_cuda_errors(cudaGetDeviceProperties(&device_prop, cuda_dev));
    printf("name......................................... = '%s'\n",
           device_prop.name);
    printf("uuid.bytes................................... = "
           "[0x%02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%"
           "02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%02hhX,0x%"
           "02hhX,0x%02hhX]\n",
           device_prop.uuid.bytes[0], device_prop.uuid.bytes[1],
           device_prop.uuid.bytes[2], device_prop.uuid.bytes[3],
           device_prop.uuid.bytes[4], device_prop.uuid.bytes[5],
           device_prop.uuid.bytes[6], device_prop.uuid.bytes[7],
           device_prop.uuid.bytes[8], device_prop.uuid.bytes[9],
           device_prop.uuid.bytes[10], device_prop.uuid.bytes[11],
           device_prop.uuid.bytes[12], device_prop.uuid.bytes[13],
           device_prop.uuid.bytes[14], device_prop.uuid.bytes[15]);
    printf("totalGlobalMem............................... = %zu\n",
           device_prop.totalGlobalMem);
    printf("sharedMemPerBlock............................ = %zu\n",
           device_prop.sharedMemPerBlock);
    printf("regsPerBlock................................. = %d\n",
           device_prop.regsPerBlock);
    printf("warpSize..................................... = %d\n",
           device_prop.warpSize);
    printf("memPitch..................................... = %zu\n",
           device_prop.memPitch);
    printf("maxThreadsPerBlock........................... = %d\n",
           device_prop.maxThreadsPerBlock);
    printf("maxThreadsDim................................ = [%d,%d,%d]\n",
           device_prop.maxThreadsDim[0], device_prop.maxThreadsDim[1],
           device_prop.maxThreadsDim[2]);
    printf("maxGridSize.................................. = [%d,%d,%d]\n",
           device_prop.maxGridSize[0], device_prop.maxGridSize[1],
           device_prop.maxGridSize[2]);
    printf("clockRate.................................... = %d\n",
           device_prop.clockRate);
    printf("totalConstMem................................ = %zu\n",
           device_prop.totalConstMem);
    printf("major........................................ = %d\n",
           device_prop.major);
    printf("minor........................................ = %d\n",
           device_prop.minor);
    printf("textureAlignment............................. = %zu\n",
           device_prop.textureAlignment);
    printf("texturePitchAlignment........................ = %zu\n",
           device_prop.texturePitchAlignment);
    printf("deviceOverlap................................ = %d\n",
           device_prop.deviceOverlap);
    printf("multiProcessorCount.......................... = %d\n",
           device_prop.multiProcessorCount);
    printf("kernelExecTimeoutEnabled..................... = %d\n",
           device_prop.kernelExecTimeoutEnabled);
    printf("integrated................................... = %d\n",
           device_prop.integrated);
    printf("canMapHostMemory............................. = %d\n",
           device_prop.canMapHostMemory);
    printf("computeMode.................................. = %d\n",
           device_prop.computeMode);
    printf("maxTexture1D................................. = %d\n",
           device_prop.maxTexture1D);
    printf("maxTexture1DMipmap........................... = %d\n",
           device_prop.maxTexture1DMipmap);
    printf("maxTexture1DLinear........................... = %d\n",
           device_prop.maxTexture1DLinear);
    printf("maxTexture2D................................. = [%d,%d]\n",
           device_prop.maxTexture2D[0], device_prop.maxTexture2D[1]);
    printf("maxTexture2DMipmap........................... = [%d,%d]\n",
           device_prop.maxTexture2DMipmap[0],
           device_prop.maxTexture2DMipmap[1]);
    printf("maxTexture2DLinear........................... = [%d,%d,%d]\n",
           device_prop.maxTexture2DLinear[0], device_prop.maxTexture2DLinear[1],
           device_prop.maxTexture2DLinear[2]);
    printf("maxTexture2DGather........................... = [%d,%d]\n",
           device_prop.maxTexture2DGather[0],
           device_prop.maxTexture2DGather[1]);
    printf("maxTexture3D................................. = [%d,%d,%d]\n",
           device_prop.maxTexture3D[0], device_prop.maxTexture3D[1],
           device_prop.maxTexture3D[2]);
    printf("maxTexture3DAlt.............................. = [%d,%d,%d]\n",
           device_prop.maxTexture3DAlt[0], device_prop.maxTexture3DAlt[1],
           device_prop.maxTexture3DAlt[2]);
    printf("maxTextureCubemap............................ = %d\n",
           device_prop.maxTextureCubemap);
    printf("maxTexture1DLayered.......................... = [%d,%d]\n",
           device_prop.maxTexture1DLayered[0],
           device_prop.maxTexture1DLayered[1]);
    printf("maxTexture2DLayered.......................... = [%d,%d,%d]\n",
           device_prop.maxTexture2DLayered[0],
           device_prop.maxTexture2DLayered[1],
           device_prop.maxTexture2DLayered[2]);
    printf("maxTextureCubemapLayered..................... = [%d,%d]\n",
           device_prop.maxTextureCubemapLayered[0],
           device_prop.maxTextureCubemapLayered[1]);
    printf("maxSurface1D................................. = %d\n",
           device_prop.maxSurface1D);
    printf("maxSurface2D................................. = [%d,%d]\n",
           device_prop.maxSurface2D[0], device_prop.maxSurface2D[1]);
    printf("maxSurface3D................................. = [%d,%d,%d]\n",
           device_prop.maxSurface3D[0], device_prop.maxSurface3D[1],
           device_prop.maxSurface3D[2]);
    printf("maxSurface1DLayered.......................... = [%d,%d]\n",
           device_prop.maxSurface1DLayered[0],
           device_prop.maxSurface1DLayered[1]);
    printf("maxSurface2DLayered.......................... = [%d,%d,%d]\n",
           device_prop.maxSurface2DLayered[0],
           device_prop.maxSurface2DLayered[1],
           device_prop.maxSurface2DLayered[2]);
    printf("maxSurfaceCubemap............................ = %d\n",
           device_prop.maxSurfaceCubemap);
    printf("maxSurfaceCubemapLayered..................... = [%d,%d]\n",
           device_prop.maxSurfaceCubemapLayered[0],
           device_prop.maxSurfaceCubemapLayered[1]);
    printf("surfaceAlignment............................. = %zu\n",
           device_prop.surfaceAlignment);
    printf("concurrentKernels............................ = %d\n",
           device_prop.concurrentKernels);
    printf("ECCEnabled................................... = %d\n",
           device_prop.ECCEnabled);
    printf("pciBusID..................................... = %d\n",
           device_prop.pciBusID);
    printf("pciDeviceID.................................. = %d\n",
           device_prop.pciDeviceID);
    printf("pciDomainID.................................. = %d\n",
           device_prop.pciDomainID);
    printf("tccDriver.................................... = %d\n",
           device_prop.tccDriver);
    printf("asyncEngineCount............................. = %d\n",
           device_prop.asyncEngineCount);
    printf("unifiedAddressing............................ = %d\n",
           device_prop.unifiedAddressing);
    printf("memoryClockRate.............................. = %d\n",
           device_prop.memoryClockRate);
    printf("memoryBusWidth............................... = %d\n",
           device_prop.memoryBusWidth);
    printf("l2CacheSize.................................. = %d\n",
           device_prop.l2CacheSize);
    printf("maxThreadsPerMultiProcessor.................. = %d\n",
           device_prop.maxThreadsPerMultiProcessor);
    printf("streamPrioritiesSupported.................... = %d\n",
           device_prop.streamPrioritiesSupported);
    printf("globalL1CacheSupported....................... = %d\n",
           device_prop.globalL1CacheSupported);
    printf("localL1CacheSupported........................ = %d\n",
           device_prop.localL1CacheSupported);
    printf("sharedMemPerMultiprocessor................... = %zu\n",
           device_prop.sharedMemPerMultiprocessor);
    printf("regsPerMultiprocessor........................ = %d\n",
           device_prop.regsPerMultiprocessor);
    printf("managedMemory................................ = %d\n",
           device_prop.managedMemory);
    printf("isMultiGpuBoard.............................. = %d\n",
           device_prop.isMultiGpuBoard);
    printf("multiGpuBoardGroupID......................... = %d\n",
           device_prop.multiGpuBoardGroupID);
    printf("singleToDoublePrecisionPerfRatio............. = %d\n",
           device_prop.singleToDoublePrecisionPerfRatio);
    printf("pageableMemoryAccess......................... = %d\n",
           device_prop.pageableMemoryAccess);
    printf("concurrentManagedAccess...................... = %d\n",
           device_prop.concurrentManagedAccess);
    printf("computePreemptionSupported................... = %d\n",
           device_prop.computePreemptionSupported);
    printf("canUseHostPointerForRegisteredMem............ = %d\n",
           device_prop.canUseHostPointerForRegisteredMem);
    printf("cooperativeLaunch............................ = %d\n",
           device_prop.cooperativeLaunch);
    printf("cooperativeMultiDeviceLaunch................. = %d\n",
           device_prop.cooperativeMultiDeviceLaunch);
    printf("pageableMemoryAccessUsesHostPageTables....... = %d\n",
           device_prop.pageableMemoryAccessUsesHostPageTables);
    printf("directManagedMemAccessFromHost............... = %d\n",
           device_prop.directManagedMemAccessFromHost);
  }
}
int main() {
  {
    int cuda_dev_number;
    check_cuda_errors(cudaGetDeviceCount(&cuda_dev_number));
    printf("cuda_dev_number = %d\n", cuda_dev_number);
    {
      int cuda_dev;
      check_cuda_errors(cudaGetDevice(&cuda_dev));
      cuda_list_attributes(cuda_dev);
      cuda_list_limits(cuda_dev);
      cuda_list_properties(cuda_dev);
    }
  }
  check_cuda_errors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  {
    int *a = static_cast<int *>(NULL);
    int *b = static_cast<int *>(NULL);
    int *c = static_cast<int *>(NULL);
    int *d_a = NULL;
    int *d_b = NULL;
    int *d_c = NULL;
    check_cuda_errors(
        cudaHostAlloc((&(a)), (N * sizeof(int)), cudaHostAllocDefault));
    check_cuda_errors(
        cudaHostAlloc((&(b)), (N * sizeof(int)), cudaHostAllocDefault));
    check_cuda_errors(
        cudaHostAlloc((&(c)), (N * sizeof(int)), cudaHostAllocDefault));
    cudaMalloc(&d_a, (N * sizeof(int)));
    cudaMalloc(&d_b, (N * sizeof(int)));
    cudaMalloc(&d_c, (N * sizeof(int)));
    for (unsigned int i = 0; (i < N); i += 1) {
      a[i] = i;
      b[i] = i;
      c[i] = 0;
    }
    cudaMemcpyAsync(d_a, a, (N * sizeof(int)), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_b, b, (N * sizeof(int)), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_c, c, (N * sizeof(int)), cudaMemcpyHostToDevice, 0);
    {
      cudaEvent_t start;
      cudaEvent_t stop;
      check_cuda_errors(cudaEventCreate(&start));
      check_cuda_errors(cudaEventCreate(&stop));
      check_cuda_errors(cudaEventRecord(start, 0));
      vector_add<<<1, N>>>(d_a, d_b, d_c, N);
      check_cuda_errors(cudaEventRecord(stop, 0));
      check_cuda_errors(cudaEventSynchronize(stop));
      {
        float time;
        check_cuda_errors(cudaEventElapsedTime(&time, start, stop));
        printf("executing kernel '(funcall vector_add<<<1,N>>> d_a d_b d_c N)' "
               "took %f ms.\n",
               time);
        check_cuda_errors(cudaEventDestroy(start));
        check_cuda_errors(cudaEventDestroy(stop));
      }
    }
    cudaMemcpyAsync(c, d_c, (N * sizeof(int)), cudaMemcpyDeviceToHost, 0);
    check_cuda_errors(cudaFreeHost(a));
    check_cuda_errors(cudaFree(d_a));
    check_cuda_errors(cudaFreeHost(b));
    check_cuda_errors(cudaFree(d_b));
    check_cuda_errors(cudaFreeHost(c));
    check_cuda_errors(cudaFree(d_c));
    {
      cuFloatComplex *fft_in_host = NULL;
      cuFloatComplex *fft_in_dev = NULL;
      auto fft_in_bytes = (NX * NY * sizeof(cuFloatComplex));
      check_cuda_errors(
          cudaHostAlloc((&(fft_in_host)), fft_in_bytes, cudaHostAllocDefault));
      check_cuda_errors(cudaMalloc((&(fft_in_dev)), fft_in_bytes));
      check_cuda_errors(cudaMemcpyAsync(fft_in_dev, fft_in_host, fft_in_bytes,
                                        cudaMemcpyHostToDevice, 0));
      {
        cudaEvent_t start;
        cudaEvent_t stop;
        check_cuda_errors(cudaEventCreate(&start));
        check_cuda_errors(cudaEventCreate(&stop));
        check_cuda_errors(cudaEventRecord(start, 0));
        fft<<<NX, NY>>>(fft_in_dev);
        check_cuda_errors(cudaEventRecord(stop, 0));
        check_cuda_errors(cudaEventSynchronize(stop));
        {
          float time;
          check_cuda_errors(cudaEventElapsedTime(&time, start, stop));
          printf("executing kernel '(funcall fft<<<NX,NY>>> fft_in_dev)' took "
                 "%f ms.\n",
                 time);
          check_cuda_errors(cudaEventDestroy(start));
          check_cuda_errors(cudaEventDestroy(stop));
        }
      }
      check_cuda_errors(cudaFreeHost(fft_in_host));
      check_cuda_errors(cudaFree(fft_in_dev));
    }
    return 0;
  }
}