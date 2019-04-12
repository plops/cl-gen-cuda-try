#include <stdio.h>
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
enum { N = 1024 };

int main() {
  {
    int cuda_dev;
    cudaChooseDevice(&cuda_dev, NULL);
    // read device attributes;
    {
      int val;
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxThreadsPerBlock, cuda_dev);
      printf("cudaDevAttrMaxThreadsPerBlock=%d (Maximum number of threads per "
             "block)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimX, cuda_dev);
      printf("cudaDevAttrMaxBlockDimX=%d (Maximum x-dimension of a block)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimY, cuda_dev);
      printf("cudaDevAttrMaxBlockDimY=%d (Maximum y-dimension of a block)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimZ, cuda_dev);
      printf("cudaDevAttrMaxBlockDimZ=%d (Maximum z-dimension of a block)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimX, cuda_dev);
      printf("cudaDevAttrMaxGridDimX=%d (Maximum x-dimension of a grid)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimY, cuda_dev);
      printf("cudaDevAttrMaxGridDimY=%d (Maximum y-dimension of a grid)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimZ, cuda_dev);
      printf("cudaDevAttrMaxGridDimZ=%d (Maximum z-dimension of a grid)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlock,
                             cuda_dev);
      printf("cudaDevAttrMaxSharedMemoryPerBlock=%d (Maximum amount of shared "
             "memoryavailable to a thread block in bytes)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrTotalConstantMemory, cuda_dev);
      printf("cudaDevAttrTotalConstantMemory=%d (Memory available on device "
             "for __constant__variables in a CUDA C kernel in bytes)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrWarpSize, cuda_dev);
      printf("cudaDevAttrWarpSize=%d (Warp size in threads)\n", val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxPitch, cuda_dev);
      printf("cudaDevAttrMaxPitch=%d (Maximum pitch in bytes allowed by the "
             "memory copyfunctions that involve memory regions allocated "
             "through cudaMallocPitch())\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture1DWidth, cuda_dev);
      printf("cudaDevAttrMaxTexture1DWidth=%d (Maximum 1D texture width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture1DLinearWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture1DLinearWidth=%d (Maximum width for a 1D "
             "texture boundto linear memory)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture1DMipmappedWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture1DMipmappedWidth=%d (Maximum mipmapped 1D "
             "texturewidth)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DWidth, cuda_dev);
      printf("cudaDevAttrMaxTexture2DWidth=%d (Maximum 2D texture width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DHeight, cuda_dev);
      printf("cudaDevAttrMaxTexture2DHeight=%d (Maximum 2D texture height)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DLinearWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DLinearWidth=%d (Maximum width for a 2D "
             "texture boundto linear memory)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DLinearHeight,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DLinearHeight=%d (Maximum height for a 2D "
             "texture boundto linear memory)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DLinearPitch,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DLinearPitch=%d (Maximum pitch in bytes "
             "for a 2D texturebound to linear memory)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DMipmappedWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DMipmappedWidth=%d (Maximum mipmapped 2D "
             "texturewidth)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DMipmappedHeight,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DMipmappedHeight=%d (Maximum mipmapped 2D "
             "textureheight)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DWidth, cuda_dev);
      printf("cudaDevAttrMaxTexture3DWidth=%d (Maximum 3D texture width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DHeight, cuda_dev);
      printf("cudaDevAttrMaxTexture3DHeight=%d (Maximum 3D texture height)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DDepth, cuda_dev);
      printf("cudaDevAttrMaxTexture3DDepth=%d (Maximum 3D texture depth)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DWidthAlt, cuda_dev);
      printf("cudaDevAttrMaxTexture3DWidthAlt=%d (Alternate maximum 3D texture "
             "width, 0 if noalternate maximum 3D texture size is supported)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DHeightAlt, cuda_dev);
      printf(
          "cudaDevAttrMaxTexture3DHeightAlt=%d (Alternate maximum 3D texture "
          "height, 0 ifno alternate maximum 3D texture size is supported)\n",
          val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DDepthAlt, cuda_dev);
      printf("cudaDevAttrMaxTexture3DDepthAlt=%d (Alternate maximum 3D texture "
             "depth, 0 if noalternate maximum 3D texture size is supported)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTextureCubemapWidth, cuda_dev);
      printf("cudaDevAttrMaxTextureCubemapWidth=%d (Maximum cubemap texture "
             "width orheight)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture1DLayeredWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture1DLayeredWidth=%d (Maximum 1D layered "
             "texture width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture1DLayeredLayers,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture1DLayeredLayers=%d (Maximum layers in a 1D "
             "layeredtexture)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DLayeredWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DLayeredWidth=%d (Maximum 2D layered "
             "texture width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DLayeredHeight,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DLayeredHeight=%d (Maximum 2D layered "
             "texture height)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DLayeredLayers,
                             cuda_dev);
      printf("cudaDevAttrMaxTexture2DLayeredLayers=%d (Maximum layers in a 2D "
             "layeredtexture)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTextureCubemapLayeredWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxTextureCubemapLayeredWidth=%d (Maximum cubemap "
             "layeredtexture width or height)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxTextureCubemapLayeredLayers,
                             cuda_dev);
      printf("cudaDevAttrMaxTextureCubemapLayeredLayers=%d (Maximum layers in "
             "a cubemaplayered texture)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface1DWidth, cuda_dev);
      printf("cudaDevAttrMaxSurface1DWidth=%d (Maximum 1D surface width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DWidth, cuda_dev);
      printf("cudaDevAttrMaxSurface2DWidth=%d (Maximum 2D surface width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DHeight, cuda_dev);
      printf("cudaDevAttrMaxSurface2DHeight=%d (Maximum 2D surface height)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DWidth, cuda_dev);
      printf("cudaDevAttrMaxSurface3DWidth=%d (Maximum 3D surface width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DHeight, cuda_dev);
      printf("cudaDevAttrMaxSurface3DHeight=%d (Maximum 3D surface height)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DDepth, cuda_dev);
      printf("cudaDevAttrMaxSurface3DDepth=%d (Maximum 3D surface depth)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface1DLayeredWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxSurface1DLayeredWidth=%d (Maximum 1D layered "
             "surface width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface1DLayeredLayers,
                             cuda_dev);
      printf("cudaDevAttrMaxSurface1DLayeredLayers=%d (Maximum layers in a 1D "
             "layeredsurface)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DLayeredWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxSurface2DLayeredWidth=%d (Maximum 2D layered "
             "surface width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DLayeredHeight,
                             cuda_dev);
      printf("cudaDevAttrMaxSurface2DLayeredHeight=%d (Maximum 2D layered "
             "surface height)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DLayeredLayers,
                             cuda_dev);
      printf("cudaDevAttrMaxSurface2DLayeredLayers=%d (Maximum layers in a 2D "
             "layeredsurface)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurfaceCubemapWidth, cuda_dev);
      printf("cudaDevAttrMaxSurfaceCubemapWidth=%d (Maximum cubemap surface "
             "width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurfaceCubemapLayeredWidth,
                             cuda_dev);
      printf("cudaDevAttrMaxSurfaceCubemapLayeredWidth=%d (Maximum cubemap "
             "layeredsurface width)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurfaceCubemapLayeredLayers,
                             cuda_dev);
      printf("cudaDevAttrMaxSurfaceCubemapLayeredLayers=%d (Maximum layers in "
             "a cubemaplayered surface)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxRegistersPerBlock, cuda_dev);
      printf("cudaDevAttrMaxRegistersPerBlock=%d (Maximum number of 32-bit "
             "registers availableto a thread block)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrClockRate, cuda_dev);
      printf("cudaDevAttrClockRate=%d (Peak clock frequency in kilohertz)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrTextureAlignment, cuda_dev);
      printf("cudaDevAttrTextureAlignment=%d (Alignment requirement texture "
             "base addressesaligned to textureAlign bytes do not need an "
             "offset applied to texture fetches)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrTexturePitchAlignment, cuda_dev);
      printf("cudaDevAttrTexturePitchAlignment=%d (Pitch alignment requirement "
             "for 2D texturereferences bound to pitched memory)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrGpuOverlap, cuda_dev);
      printf("cudaDevAttrGpuOverlap=%d (1 if the device can concurrently copy "
             "memory betweenhost and device while executing a kernel, or 0 if "
             "not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount, cuda_dev);
      printf("cudaDevAttrMultiProcessorCount=%d (Number of multiprocessors on "
             "the device)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrKernelExecTimeout, cuda_dev);
      printf("cudaDevAttrKernelExecTimeout=%d (1 if there is a run time limit "
             "for kernels executedon the device, or 0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, cuda_dev);
      printf("cudaDevAttrIntegrated=%d (1 if the device is integrated with the "
             "memory subsystem, or0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrCanMapHostMemory, cuda_dev);
      printf("cudaDevAttrCanMapHostMemory=%d (1 if the device can map host "
             "memory into theCUDA address space, or 0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrComputeMode, cuda_dev);
      printf("cudaDevAttrComputeMode=%d (Compute mode is the compute mode that "
             "the device iscurrently in. Available modes are as follows)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrConcurrentKernels, cuda_dev);
      printf("cudaDevAttrConcurrentKernels=%d (1 if the device supports "
             "executing multiple kernelswithin the same context "
             "simultaneously, or 0 if not. It is not guaranteed that "
             "multipkernels will be resident on the device concurrently so "
             "this feature should not berelied upon for correctness)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrEccEnabled, cuda_dev);
      printf("cudaDevAttrEccEnabled=%d (1 if error correction is enabled on "
             "the device, 0 if errorcorrection is disabled or not supported by "
             "the device)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrPciBusId, cuda_dev);
      printf("cudaDevAttrPciBusId=%d (PCI bus identifier of the device)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrPciDeviceId, cuda_dev);
      printf("cudaDevAttrPciDeviceId=%d (PCI device (also known as slot) "
             "identifier of the device)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrTccDriver, cuda_dev);
      printf("cudaDevAttrTccDriver=%d (1 if the device is using a TCC driver. "
             "TCC is only availableon Tesla hardware running Windows Vista or "
             "later)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMemoryClockRate, cuda_dev);
      printf("cudaDevAttrMemoryClockRate=%d (Peak memory clock frequency in "
             "kilohertz)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrGlobalMemoryBusWidth, cuda_dev);
      printf("cudaDevAttrGlobalMemoryBusWidth=%d (Global memory bus width in "
             "bits)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrL2CacheSize, cuda_dev);
      printf("cudaDevAttrL2CacheSize=%d (Size of L2 cache in bytes. 0 if the "
             "device doesn't have L2cache)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxThreadsPerMultiProcessor,
                             cuda_dev);
      printf("cudaDevAttrMaxThreadsPerMultiProcessor=%d (Maximum resident "
             "threads permultiprocessor)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrUnifiedAddressing, cuda_dev);
      printf("cudaDevAttrUnifiedAddressing=%d (1 if the device shares a "
             "unified address space withthe host, or 0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrComputeCapabilityMajor, cuda_dev);
      printf("cudaDevAttrComputeCapabilityMajor=%d (Major compute capability "
             "version number)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrComputeCapabilityMinor, cuda_dev);
      printf("cudaDevAttrComputeCapabilityMinor=%d (Minor compute capability "
             "version number)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrStreamPrioritiesSupported,
                             cuda_dev);
      printf("cudaDevAttrStreamPrioritiesSupported=%d (1 if the device "
             "supports stream priorities, or0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrGlobalL1CacheSupported, cuda_dev);
      printf("cudaDevAttrGlobalL1CacheSupported=%d (1 if device supports "
             "caching globals in L1cache, 0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrLocalL1CacheSupported, cuda_dev);
      printf("cudaDevAttrLocalL1CacheSupported=%d (1 if device supports "
             "caching locals in L1cache, 0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                             cuda_dev);
      printf("cudaDevAttrMaxSharedMemoryPerMultiprocessor=%d (Maximum amount "
             "of sharedmemory available to a multiprocessor in bytes this "
             "amount is shared by all threadblocks simultaneously resident on "
             "a multiprocessor)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxRegistersPerMultiprocessor,
                             cuda_dev);
      printf(
          "cudaDevAttrMaxRegistersPerMultiprocessor=%d (Maximum number of "
          "32-bitregisters available to a multiprocessor this number is shared "
          "by all thread blockssimultaneously resident on a multiprocessor)\n",
          val);
      cudaDeviceGetAttribute(&val, cudaDevAttrManagedMemory, cuda_dev);
      printf("cudaDevAttrManagedMemory=%d (1 if device supports allocating "
             "managed memory, 0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrIsMultiGpuBoard, cuda_dev);
      printf("cudaDevAttrIsMultiGpuBoard=%d (1 if device is on a multi-GPU "
             "board, 0 if not)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMultiGpuBoardGroupID, cuda_dev);
      printf("cudaDevAttrMultiGpuBoardGroupID=%d (Unique identifier for a "
             "group of devices onthe same multi-GPU board)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrHostNativeAtomicSupported,
                             cuda_dev);
      printf("cudaDevAttrHostNativeAtomicSupported=%d (1 if the link between "
             "the device and thehost supports native atomic operations)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrSingleToDoublePrecisionPerfRatio,
                             cuda_dev);
      printf("cudaDevAttrSingleToDoublePrecisionPerfRatio=%d (Ratio of single "
             "precisionperformance (in floating-point operations per second) "
             "to double precision performance)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrPageableMemoryAccess, cuda_dev);
      printf("cudaDevAttrPageableMemoryAccess=%d (1 if the device supports "
             "coherently accessingpageable memory without calling "
             "cudaHostRegister on it, and 0 otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrConcurrentManagedAccess,
                             cuda_dev);
      printf("cudaDevAttrConcurrentManagedAccess=%d (1 if the device can "
             "coherently accessmanaged memory concurrently with the CPU, and 0 "
             "otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrComputePreemptionSupported,
                             cuda_dev);
      printf("cudaDevAttrComputePreemptionSupported=%d (1 if the device "
             "supports ComputePreemption, 0 if not.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrCanUseHostPointerForRegisteredMem,
                             cuda_dev);
      printf("cudaDevAttrCanUseHostPointerForRegisteredMem=%d (1 if the device "
             "can access hostregistered memory at the same virtual address as "
             "the CPU, and 0 otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrCooperativeLaunch, cuda_dev);
      printf("cudaDevAttrCooperativeLaunch=%d (1 if the device supports "
             "launching cooperativekernels via cudaLaunchCooperativeKernel, "
             "and 0 otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrCooperativeMultiDeviceLaunch,
                             cuda_dev);
      printf("cudaDevAttrCooperativeMultiDeviceLaunch=%d (1 if the device "
             "supports launchingcooperative kernels via "
             "cudaLaunchCooperativeKernelMultiDevice, and 0otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrCanFlushRemoteWrites, cuda_dev);
      printf("cudaDevAttrCanFlushRemoteWrites=%d (1 if the device supports "
             "flushing ofoutstanding remote writes, and 0 otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrHostRegisterSupported, cuda_dev);
      printf(
          "cudaDevAttrHostRegisterSupported=%d (1 if the device supports host "
          "memoryregistration via cudaHostRegister, and 0 otherwise.)\n",
          val);
      cudaDeviceGetAttribute(
          &val, cudaDevAttrPageableMemoryAccessUsesHostPageTables, cuda_dev);
      printf("cudaDevAttrPageableMemoryAccessUsesHostPageTables=%d (1 if the "
             "device accessespageable memory via the host's page tables, and 0 "
             "otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrDirectManagedMemAccessFromHost,
                             cuda_dev);
      printf("cudaDevAttrDirectManagedMemAccessFromHost=%d (1 if the host can "
             "directly accessmanaged memory on the device without migration, "
             "and 0 otherwise.)\n",
             val);
      cudaDeviceGetAttribute(&val, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                             cuda_dev);
      printf("cudaDevAttrMaxSharedMemoryPerBlockOptin=%d (Maximum per "
             "blockshared memory size on the device. This value can be opted "
             "into when usingcudaFuncSetAttribute)\n",
             val);
    }
  }
  {
    int *a = static_cast<int *>(malloc((N * sizeof(int))));
    int *b = static_cast<int *>(malloc((N * sizeof(int))));
    int *c = static_cast<int *>(malloc((N * sizeof(int))));
    int *d_a;
    int *d_b;
    int *d_c;
    cudaMalloc(&d_a, (N * sizeof(int)));
    cudaMalloc(&d_b, (N * sizeof(int)));
    cudaMalloc(&d_c, (N * sizeof(int)));
    for (unsigned int i = 0; (i < N); i += 1) {
      a[i] = i;
      b[i] = i;
      c[i] = 0;
    }
    cudaMemcpy(d_a, a, (N * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, (N * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, (N * sizeof(int)), cudaMemcpyHostToDevice);
    vector_add<<<1, N>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c, d_c, (N * sizeof(int)), cudaMemcpyDeviceToHost);
    free(a);
    cudaFree(d_a);
    free(b);
    cudaFree(d_b);
    free(c);
    cudaFree(d_c);
    return 0;
  }
}