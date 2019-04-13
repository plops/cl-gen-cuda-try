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
enum { N = 1024, NX = 256, NY = 256 };

__global__ void fft(cuFloatComplex *__restrict__ in);
void cuda_list_attributes(int cuda_dev) {
  {
    int val;
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxThreadsPerBlock, cuda_dev));
    printf("cudaDevAttrMaxThreadsPerBlock=%d (Maximum number of threads per "
           "block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimX, cuda_dev));
    printf("cudaDevAttrMaxBlockDimX=%d (Maximum x-dimension of a block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimY, cuda_dev));
    printf("cudaDevAttrMaxBlockDimY=%d (Maximum y-dimension of a block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxBlockDimZ, cuda_dev));
    printf("cudaDevAttrMaxBlockDimZ=%d (Maximum z-dimension of a block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimX, cuda_dev));
    printf("cudaDevAttrMaxGridDimX=%d (Maximum x-dimension of a grid)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimY, cuda_dev));
    printf("cudaDevAttrMaxGridDimY=%d (Maximum y-dimension of a grid)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxGridDimZ, cuda_dev));
    printf("cudaDevAttrMaxGridDimZ=%d (Maximum z-dimension of a grid)\n", val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSharedMemoryPerBlock, cuda_dev));
    printf("cudaDevAttrMaxSharedMemoryPerBlock=%d (Maximum amount of shared "
           "memoryavailable to a thread block in bytes)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrTotalConstantMemory, cuda_dev));
    printf("cudaDevAttrTotalConstantMemory=%d (Memory available on device for "
           "__constant__variables in a CUDA C kernel in bytes)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrWarpSize, cuda_dev));
    printf("cudaDevAttrWarpSize=%d (Warp size in threads)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxPitch, cuda_dev));
    printf("cudaDevAttrMaxPitch=%d (Maximum pitch in bytes allowed by the "
           "memory copyfunctions that involve memory regions allocated through "
           "cudaMallocPitch())\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture1DWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DWidth=%d (Maximum 1D texture width)\n", val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DLinearWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DLinearWidth=%d (Maximum width for a 1D "
           "texture boundto linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DMipmappedWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DMipmappedWidth=%d (Maximum mipmapped 1D "
           "texturewidth)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DWidth=%d (Maximum 2D texture width)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture2DHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DHeight=%d (Maximum 2D texture height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLinearWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLinearWidth=%d (Maximum width for a 2D "
           "texture boundto linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLinearHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLinearHeight=%d (Maximum height for a 2D "
           "texture boundto linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLinearPitch, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLinearPitch=%d (Maximum pitch in bytes for "
           "a 2D texturebound to linear memory)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DMipmappedWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DMipmappedWidth=%d (Maximum mipmapped 2D "
           "texturewidth)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DMipmappedHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DMipmappedHeight=%d (Maximum mipmapped 2D "
           "textureheight)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture3DWidth=%d (Maximum 3D texture width)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture3DHeight=%d (Maximum 3D texture height)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxTexture3DDepth, cuda_dev));
    printf("cudaDevAttrMaxTexture3DDepth=%d (Maximum 3D texture depth)\n", val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture3DWidthAlt, cuda_dev));
    printf("cudaDevAttrMaxTexture3DWidthAlt=%d (Alternate maximum 3D texture "
           "width, 0 if noalternate maximum 3D texture size is supported)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture3DHeightAlt, cuda_dev));
    printf("cudaDevAttrMaxTexture3DHeightAlt=%d (Alternate maximum 3D texture "
           "height, 0 ifno alternate maximum 3D texture size is supported)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture3DDepthAlt, cuda_dev));
    printf("cudaDevAttrMaxTexture3DDepthAlt=%d (Alternate maximum 3D texture "
           "depth, 0 if noalternate maximum 3D texture size is supported)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTextureCubemapWidth, cuda_dev));
    printf("cudaDevAttrMaxTextureCubemapWidth=%d (Maximum cubemap texture "
           "width orheight)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture1DLayeredWidth=%d (Maximum 1D layered texture "
           "width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture1DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxTexture1DLayeredLayers=%d (Maximum layers in a 1D "
           "layeredtexture)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLayeredWidth=%d (Maximum 2D layered texture "
           "width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLayeredHeight, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLayeredHeight=%d (Maximum 2D layered "
           "texture height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTexture2DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxTexture2DLayeredLayers=%d (Maximum layers in a 2D "
           "layeredtexture)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTextureCubemapLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxTextureCubemapLayeredWidth=%d (Maximum cubemap "
           "layeredtexture width or height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxTextureCubemapLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxTextureCubemapLayeredLayers=%d (Maximum layers in a "
           "cubemaplayered texture)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface1DWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface1DWidth=%d (Maximum 1D surface width)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface2DWidth=%d (Maximum 2D surface width)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface2DHeight, cuda_dev));
    printf("cudaDevAttrMaxSurface2DHeight=%d (Maximum 2D surface height)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface3DWidth=%d (Maximum 3D surface width)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DHeight, cuda_dev));
    printf("cudaDevAttrMaxSurface3DHeight=%d (Maximum 3D surface height)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMaxSurface3DDepth, cuda_dev));
    printf("cudaDevAttrMaxSurface3DDepth=%d (Maximum 3D surface depth)\n", val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface1DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface1DLayeredWidth=%d (Maximum 1D layered surface "
           "width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface1DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxSurface1DLayeredLayers=%d (Maximum layers in a 1D "
           "layeredsurface)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface2DLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxSurface2DLayeredWidth=%d (Maximum 2D layered surface "
           "width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface2DLayeredHeight, cuda_dev));
    printf("cudaDevAttrMaxSurface2DLayeredHeight=%d (Maximum 2D layered "
           "surface height)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurface2DLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxSurface2DLayeredLayers=%d (Maximum layers in a 2D "
           "layeredsurface)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurfaceCubemapWidth, cuda_dev));
    printf("cudaDevAttrMaxSurfaceCubemapWidth=%d (Maximum cubemap surface "
           "width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurfaceCubemapLayeredWidth, cuda_dev));
    printf("cudaDevAttrMaxSurfaceCubemapLayeredWidth=%d (Maximum cubemap "
           "layeredsurface width)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSurfaceCubemapLayeredLayers, cuda_dev));
    printf("cudaDevAttrMaxSurfaceCubemapLayeredLayers=%d (Maximum layers in a "
           "cubemaplayered surface)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxRegistersPerBlock, cuda_dev));
    printf("cudaDevAttrMaxRegistersPerBlock=%d (Maximum number of 32-bit "
           "registers availableto a thread block)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrClockRate, cuda_dev));
    printf("cudaDevAttrClockRate=%d (Peak clock frequency in kilohertz)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrTextureAlignment, cuda_dev));
    printf("cudaDevAttrTextureAlignment=%d (Alignment requirement texture base "
           "addressesaligned to textureAlign bytes do not need an offset "
           "applied to texture fetches)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrTexturePitchAlignment, cuda_dev));
    printf("cudaDevAttrTexturePitchAlignment=%d (Pitch alignment requirement "
           "for 2D texturereferences bound to pitched memory)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrGpuOverlap, cuda_dev));
    printf("cudaDevAttrGpuOverlap=%d (1 if the device can concurrently copy "
           "memory betweenhost and device while executing a kernel, or 0 if "
           "not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMultiProcessorCount, cuda_dev));
    printf("cudaDevAttrMultiProcessorCount=%d (Number of multiprocessors on "
           "the device)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrKernelExecTimeout, cuda_dev));
    printf("cudaDevAttrKernelExecTimeout=%d (1 if there is a run time limit "
           "for kernels executedon the device, or 0 if not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrIntegrated, cuda_dev));
    printf("cudaDevAttrIntegrated=%d (1 if the device is integrated with the "
           "memory subsystem, or0 if not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrCanMapHostMemory, cuda_dev));
    printf("cudaDevAttrCanMapHostMemory=%d (1 if the device can map host "
           "memory into theCUDA address space, or 0 if not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrComputeMode, cuda_dev));
    printf("cudaDevAttrComputeMode=%d (Compute mode is the compute mode that "
           "the device iscurrently in. Available modes are as follows)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrConcurrentKernels, cuda_dev));
    printf("cudaDevAttrConcurrentKernels=%d (1 if the device supports "
           "executing multiple kernelswithin the same context simultaneously, "
           "or 0 if not. It is not guaranteed that multipkernels will be "
           "resident on the device concurrently so this feature should not "
           "berelied upon for correctness)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrEccEnabled, cuda_dev));
    printf("cudaDevAttrEccEnabled=%d (1 if error correction is enabled on the "
           "device, 0 if errorcorrection is disabled or not supported by the "
           "device)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrPciBusId, cuda_dev));
    printf("cudaDevAttrPciBusId=%d (PCI bus identifier of the device)\n", val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrPciDeviceId, cuda_dev));
    printf("cudaDevAttrPciDeviceId=%d (PCI device (also known as slot) "
           "identifier of the device)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrTccDriver, cuda_dev));
    printf(
        "cudaDevAttrTccDriver=%d (1 if the device is using a TCC driver. TCC "
        "is only availableon Tesla hardware running Windows Vista or later)\n",
        val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrMemoryClockRate, cuda_dev));
    printf("cudaDevAttrMemoryClockRate=%d (Peak memory clock frequency in "
           "kilohertz)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrGlobalMemoryBusWidth, cuda_dev));
    printf("cudaDevAttrGlobalMemoryBusWidth=%d (Global memory bus width in "
           "bits)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrL2CacheSize, cuda_dev));
    printf("cudaDevAttrL2CacheSize=%d (Size of L2 cache in bytes. 0 if the "
           "device doesn't have L2cache)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxThreadsPerMultiProcessor, cuda_dev));
    printf("cudaDevAttrMaxThreadsPerMultiProcessor=%d (Maximum resident "
           "threads permultiprocessor)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrUnifiedAddressing, cuda_dev));
    printf("cudaDevAttrUnifiedAddressing=%d (1 if the device shares a unified "
           "address space withthe host, or 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrComputeCapabilityMajor, cuda_dev));
    printf("cudaDevAttrComputeCapabilityMajor=%d (Major compute capability "
           "version number)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrComputeCapabilityMinor, cuda_dev));
    printf("cudaDevAttrComputeCapabilityMinor=%d (Minor compute capability "
           "version number)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrStreamPrioritiesSupported, cuda_dev));
    printf("cudaDevAttrStreamPrioritiesSupported=%d (1 if the device supports "
           "stream priorities, or0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrGlobalL1CacheSupported, cuda_dev));
    printf("cudaDevAttrGlobalL1CacheSupported=%d (1 if device supports caching "
           "globals in L1cache, 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrLocalL1CacheSupported, cuda_dev));
    printf("cudaDevAttrLocalL1CacheSupported=%d (1 if device supports caching "
           "locals in L1cache, 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSharedMemoryPerMultiprocessor, cuda_dev));
    printf("cudaDevAttrMaxSharedMemoryPerMultiprocessor=%d (Maximum amount of "
           "sharedmemory available to a multiprocessor in bytes this amount is "
           "shared by all threadblocks simultaneously resident on a "
           "multiprocessor)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxRegistersPerMultiprocessor, cuda_dev));
    printf(
        "cudaDevAttrMaxRegistersPerMultiprocessor=%d (Maximum number of "
        "32-bitregisters available to a multiprocessor this number is shared "
        "by all thread blockssimultaneously resident on a multiprocessor)\n",
        val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrManagedMemory, cuda_dev));
    printf("cudaDevAttrManagedMemory=%d (1 if device supports allocating "
           "managed memory, 0 if not)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrIsMultiGpuBoard, cuda_dev));
    printf("cudaDevAttrIsMultiGpuBoard=%d (1 if device is on a multi-GPU "
           "board, 0 if not)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMultiGpuBoardGroupID, cuda_dev));
    printf("cudaDevAttrMultiGpuBoardGroupID=%d (Unique identifier for a group "
           "of devices onthe same multi-GPU board)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrHostNativeAtomicSupported, cuda_dev));
    printf("cudaDevAttrHostNativeAtomicSupported=%d (1 if the link between the "
           "device and thehost supports native atomic operations)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrSingleToDoublePrecisionPerfRatio, cuda_dev));
    printf("cudaDevAttrSingleToDoublePrecisionPerfRatio=%d (Ratio of single "
           "precisionperformance (in floating-point operations per second) to "
           "double precision performance)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrPageableMemoryAccess, cuda_dev));
    printf("cudaDevAttrPageableMemoryAccess=%d (1 if the device supports "
           "coherently accessingpageable memory without calling "
           "cudaHostRegister on it, and 0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrConcurrentManagedAccess, cuda_dev));
    printf(
        "cudaDevAttrConcurrentManagedAccess=%d (1 if the device can coherently "
        "accessmanaged memory concurrently with the CPU, and 0 otherwise.)\n",
        val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrComputePreemptionSupported, cuda_dev));
    printf("cudaDevAttrComputePreemptionSupported=%d (1 if the device supports "
           "ComputePreemption, 0 if not.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrCanUseHostPointerForRegisteredMem, cuda_dev));
    printf("cudaDevAttrCanUseHostPointerForRegisteredMem=%d (1 if the device "
           "can access hostregistered memory at the same virtual address as "
           "the CPU, and 0 otherwise.)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetAttribute(&val, cudaDevAttrCooperativeLaunch, cuda_dev));
    printf("cudaDevAttrCooperativeLaunch=%d (1 if the device supports "
           "launching cooperativekernels via cudaLaunchCooperativeKernel, and "
           "0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrCooperativeMultiDeviceLaunch, cuda_dev));
    printf("cudaDevAttrCooperativeMultiDeviceLaunch=%d (1 if the device "
           "supports launchingcooperative kernels via "
           "cudaLaunchCooperativeKernelMultiDevice, and 0otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrCanFlushRemoteWrites, cuda_dev));
    printf("cudaDevAttrCanFlushRemoteWrites=%d (1 if the device supports "
           "flushing ofoutstanding remote writes, and 0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrHostRegisterSupported, cuda_dev));
    printf("cudaDevAttrHostRegisterSupported=%d (1 if the device supports host "
           "memoryregistration via cudaHostRegister, and 0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrPageableMemoryAccessUsesHostPageTables, cuda_dev));
    printf("cudaDevAttrPageableMemoryAccessUsesHostPageTables=%d (1 if the "
           "device accessespageable memory via the host's page tables, and 0 "
           "otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrDirectManagedMemAccessFromHost, cuda_dev));
    printf("cudaDevAttrDirectManagedMemAccessFromHost=%d (1 if the host can "
           "directly accessmanaged memory on the device without migration, and "
           "0 otherwise.)\n",
           val);
    check_cuda_errors(cudaDeviceGetAttribute(
        &val, cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda_dev));
    printf("cudaDevAttrMaxSharedMemoryPerBlockOptin=%d (Maximum per "
           "blockshared memory size on the device. This value can be opted "
           "into when usingcudaFuncSetAttribute)\n",
           val);
  }
}
void cuda_list_limits(int cuda_dev) {
  {
    size_t val;
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitStackSize));
    printf("cudaLimitStackSize=%lu (stack size in bytes of each GPU thread)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitPrintfFifoSize));
    printf("cudaLimitPrintfFifoSize=%lu (size in bytes of the shared FIFO used "
           "by the printf() devicesystem call)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitMallocHeapSize));
    printf("cudaLimitMallocHeapSize=%lu (size in bytes of the heap used by the "
           "malloc() and free()device system calls)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitDevRuntimeSyncDepth));
    printf("cudaLimitDevRuntimeSyncDepth=%lu (maximum grid depth at which a "
           "thread canisssue the device runtime call cudaDeviceSynchronize() "
           "to wait on child gridlaunches to complete.)\n",
           val);
    check_cuda_errors(
        cudaDeviceGetLimit(&val, cudaLimitDevRuntimePendingLaunchCount));
    printf("cudaLimitDevRuntimePendingLaunchCount=%lu (maximum number of "
           "outstandingdevice runtime launches.)\n",
           val);
    check_cuda_errors(cudaDeviceGetLimit(&val, cudaLimitMaxL2FetchGranularity));
    printf("cudaLimitMaxL2FetchGranularity=%lu (L2 cache fetch granularity.)\n",
           val);
  }
}
void cuda_list_properties(int cuda_dev) {
  {
    cudaDeviceProp device_prop;
    check_cuda_errors(cudaGetDeviceProperties(&device_prop, cuda_dev));
    printf("name = "
           "[%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,"
           "%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%"
           "c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c,%c]\n",
           device_prop.name[0], device_prop.name[1], device_prop.name[2],
           device_prop.name[3], device_prop.name[4], device_prop.name[5],
           device_prop.name[6], device_prop.name[7], device_prop.name[8],
           device_prop.name[9], device_prop.name[10], device_prop.name[11],
           device_prop.name[12], device_prop.name[13], device_prop.name[14],
           device_prop.name[15], device_prop.name[16], device_prop.name[17],
           device_prop.name[18], device_prop.name[19], device_prop.name[20],
           device_prop.name[21], device_prop.name[22], device_prop.name[23],
           device_prop.name[24], device_prop.name[25], device_prop.name[26],
           device_prop.name[27], device_prop.name[28], device_prop.name[29],
           device_prop.name[30], device_prop.name[31], device_prop.name[32],
           device_prop.name[33], device_prop.name[34], device_prop.name[35],
           device_prop.name[36], device_prop.name[37], device_prop.name[38],
           device_prop.name[39], device_prop.name[40], device_prop.name[41],
           device_prop.name[42], device_prop.name[43], device_prop.name[44],
           device_prop.name[45], device_prop.name[46], device_prop.name[47],
           device_prop.name[48], device_prop.name[49], device_prop.name[50],
           device_prop.name[51], device_prop.name[52], device_prop.name[53],
           device_prop.name[54], device_prop.name[55], device_prop.name[56],
           device_prop.name[57], device_prop.name[58], device_prop.name[59],
           device_prop.name[60], device_prop.name[61], device_prop.name[62],
           device_prop.name[63], device_prop.name[64], device_prop.name[65],
           device_prop.name[66], device_prop.name[67], device_prop.name[68],
           device_prop.name[69], device_prop.name[70], device_prop.name[71],
           device_prop.name[72], device_prop.name[73], device_prop.name[74],
           device_prop.name[75], device_prop.name[76], device_prop.name[77],
           device_prop.name[78], device_prop.name[79], device_prop.name[80],
           device_prop.name[81], device_prop.name[82], device_prop.name[83],
           device_prop.name[84], device_prop.name[85], device_prop.name[86],
           device_prop.name[87], device_prop.name[88], device_prop.name[89],
           device_prop.name[90], device_prop.name[91], device_prop.name[92],
           device_prop.name[93], device_prop.name[94], device_prop.name[95],
           device_prop.name[96], device_prop.name[97], device_prop.name[98],
           device_prop.name[99], device_prop.name[100], device_prop.name[101],
           device_prop.name[102], device_prop.name[103], device_prop.name[104],
           device_prop.name[105], device_prop.name[106], device_prop.name[107],
           device_prop.name[108], device_prop.name[109], device_prop.name[110],
           device_prop.name[111], device_prop.name[112], device_prop.name[113],
           device_prop.name[114], device_prop.name[115], device_prop.name[116],
           device_prop.name[117], device_prop.name[118], device_prop.name[119],
           device_prop.name[120], device_prop.name[121], device_prop.name[122],
           device_prop.name[123], device_prop.name[124], device_prop.name[125],
           device_prop.name[126], device_prop.name[127], device_prop.name[128],
           device_prop.name[129], device_prop.name[130], device_prop.name[131],
           device_prop.name[132], device_prop.name[133], device_prop.name[134],
           device_prop.name[135], device_prop.name[136], device_prop.name[137],
           device_prop.name[138], device_prop.name[139], device_prop.name[140],
           device_prop.name[141], device_prop.name[142], device_prop.name[143],
           device_prop.name[144], device_prop.name[145], device_prop.name[146],
           device_prop.name[147], device_prop.name[148], device_prop.name[149],
           device_prop.name[150], device_prop.name[151], device_prop.name[152],
           device_prop.name[153], device_prop.name[154], device_prop.name[155],
           device_prop.name[156], device_prop.name[157], device_prop.name[158],
           device_prop.name[159], device_prop.name[160], device_prop.name[161],
           device_prop.name[162], device_prop.name[163], device_prop.name[164],
           device_prop.name[165], device_prop.name[166], device_prop.name[167],
           device_prop.name[168], device_prop.name[169], device_prop.name[170],
           device_prop.name[171], device_prop.name[172], device_prop.name[173],
           device_prop.name[174], device_prop.name[175], device_prop.name[176],
           device_prop.name[177], device_prop.name[178], device_prop.name[179],
           device_prop.name[180], device_prop.name[181], device_prop.name[182],
           device_prop.name[183], device_prop.name[184], device_prop.name[185],
           device_prop.name[186], device_prop.name[187], device_prop.name[188],
           device_prop.name[189], device_prop.name[190], device_prop.name[191],
           device_prop.name[192], device_prop.name[193], device_prop.name[194],
           device_prop.name[195], device_prop.name[196], device_prop.name[197],
           device_prop.name[198], device_prop.name[199], device_prop.name[200],
           device_prop.name[201], device_prop.name[202], device_prop.name[203],
           device_prop.name[204], device_prop.name[205], device_prop.name[206],
           device_prop.name[207], device_prop.name[208], device_prop.name[209],
           device_prop.name[210], device_prop.name[211], device_prop.name[212],
           device_prop.name[213], device_prop.name[214], device_prop.name[215],
           device_prop.name[216], device_prop.name[217], device_prop.name[218],
           device_prop.name[219], device_prop.name[220], device_prop.name[221],
           device_prop.name[222], device_prop.name[223], device_prop.name[224],
           device_prop.name[225], device_prop.name[226], device_prop.name[227],
           device_prop.name[228], device_prop.name[229], device_prop.name[230],
           device_prop.name[231], device_prop.name[232], device_prop.name[233],
           device_prop.name[234], device_prop.name[235], device_prop.name[236],
           device_prop.name[237], device_prop.name[238], device_prop.name[239],
           device_prop.name[240], device_prop.name[241], device_prop.name[242],
           device_prop.name[243], device_prop.name[244], device_prop.name[245],
           device_prop.name[246], device_prop.name[247], device_prop.name[248],
           device_prop.name[249], device_prop.name[250], device_prop.name[251],
           device_prop.name[252], device_prop.name[253], device_prop.name[254],
           device_prop.name[255]);
    printf("uuid.bytes = "
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
    printf("totalGlobalMem = %zu\n", device_prop.totalGlobalMem);
    printf("sharedMemPerBlock = %zu\n", device_prop.sharedMemPerBlock);
    printf("regsPerBlock = %d\n", device_prop.regsPerBlock);
    printf("warpSize = %d\n", device_prop.warpSize);
    printf("memPitch = %zu\n", device_prop.memPitch);
    printf("maxThreadsPerBlock = %d\n", device_prop.maxThreadsPerBlock);
    printf("maxThreadsDim = [%d,%d,%d]\n", device_prop.maxThreadsDim[0],
           device_prop.maxThreadsDim[1], device_prop.maxThreadsDim[2]);
    printf("maxGridSize = [%d,%d,%d]\n", device_prop.maxGridSize[0],
           device_prop.maxGridSize[1], device_prop.maxGridSize[2]);
    printf("clockRate = %d\n", device_prop.clockRate);
    printf("totalConstMem = %zu\n", device_prop.totalConstMem);
    printf("major = %d\n", device_prop.major);
    printf("minor = %d\n", device_prop.minor);
    printf("textureAlignment = %zu\n", device_prop.textureAlignment);
    printf("texturePitchAlignment = %zu\n", device_prop.texturePitchAlignment);
    printf("deviceOverlap = %d\n", device_prop.deviceOverlap);
    printf("multiProcessorCount = %d\n", device_prop.multiProcessorCount);
    printf("kernelExecTimeoutEnabled = %d\n",
           device_prop.kernelExecTimeoutEnabled);
    printf("integrated = %d\n", device_prop.integrated);
    printf("canMapHostMemory = %d\n", device_prop.canMapHostMemory);
    printf("computeMode = %d\n", device_prop.computeMode);
    printf("maxTexture1D = %d\n", device_prop.maxTexture1D);
    printf("maxTexture1DMipmap = %d\n", device_prop.maxTexture1DMipmap);
    printf("maxTexture1DLinear = %d\n", device_prop.maxTexture1DLinear);
    printf("maxTexture2D = [%d,%d]\n", device_prop.maxTexture2D[0],
           device_prop.maxTexture2D[1]);
    printf("maxTexture2DMipmap = [%d,%d]\n", device_prop.maxTexture2DMipmap[0],
           device_prop.maxTexture2DMipmap[1]);
    printf("maxTexture2DLinear = [%d,%d,%d]\n",
           device_prop.maxTexture2DLinear[0], device_prop.maxTexture2DLinear[1],
           device_prop.maxTexture2DLinear[2]);
    printf("maxTexture2DGather = [%d,%d]\n", device_prop.maxTexture2DGather[0],
           device_prop.maxTexture2DGather[1]);
    printf("maxTexture3D = [%d,%d,%d]\n", device_prop.maxTexture3D[0],
           device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
    printf("maxTexture3DAlt = [%d,%d,%d]\n", device_prop.maxTexture3DAlt[0],
           device_prop.maxTexture3DAlt[1], device_prop.maxTexture3DAlt[2]);
    printf("maxTextureCubemap = %d\n", device_prop.maxTextureCubemap);
    printf("maxTexture1DLayered = [%d,%d]\n",
           device_prop.maxTexture1DLayered[0],
           device_prop.maxTexture1DLayered[1]);
    printf("maxTexture2DLayered = [%d,%d,%d]\n",
           device_prop.maxTexture2DLayered[0],
           device_prop.maxTexture2DLayered[1],
           device_prop.maxTexture2DLayered[2]);
    printf("maxTextureCubemapLayered = [%d,%d]\n",
           device_prop.maxTextureCubemapLayered[0],
           device_prop.maxTextureCubemapLayered[1]);
    printf("maxSurface1D = %d\n", device_prop.maxSurface1D);
    printf("maxSurface2D = [%d,%d]\n", device_prop.maxSurface2D[0],
           device_prop.maxSurface2D[1]);
    printf("maxSurface3D = [%d,%d,%d]\n", device_prop.maxSurface3D[0],
           device_prop.maxSurface3D[1], device_prop.maxSurface3D[2]);
    printf("maxSurface1DLayered = [%d,%d]\n",
           device_prop.maxSurface1DLayered[0],
           device_prop.maxSurface1DLayered[1]);
    printf("maxSurface2DLayered = [%d,%d,%d]\n",
           device_prop.maxSurface2DLayered[0],
           device_prop.maxSurface2DLayered[1],
           device_prop.maxSurface2DLayered[2]);
    printf("maxSurfaceCubemap = %d\n", device_prop.maxSurfaceCubemap);
    printf("maxSurfaceCubemapLayered = [%d,%d]\n",
           device_prop.maxSurfaceCubemapLayered[0],
           device_prop.maxSurfaceCubemapLayered[1]);
    printf("surfaceAlignment = %zu\n", device_prop.surfaceAlignment);
    printf("concurrentKernels = %d\n", device_prop.concurrentKernels);
    printf("ECCEnabled = %d\n", device_prop.ECCEnabled);
    printf("pciBusID = %d\n", device_prop.pciBusID);
    printf("pciDeviceID = %d\n", device_prop.pciDeviceID);
    printf("pciDomainID = %d\n", device_prop.pciDomainID);
    printf("tccDriver = %d\n", device_prop.tccDriver);
    printf("asyncEngineCount = %d\n", device_prop.asyncEngineCount);
    printf("unifiedAddressing = %d\n", device_prop.unifiedAddressing);
    printf("memoryClockRate = %d\n", device_prop.memoryClockRate);
    printf("memoryBusWidth = %d\n", device_prop.memoryBusWidth);
    printf("l2CacheSize = %d\n", device_prop.l2CacheSize);
    printf("maxThreadsPerMultiProcessor = %d\n",
           device_prop.maxThreadsPerMultiProcessor);
    printf("streamPrioritiesSupported = %d\n",
           device_prop.streamPrioritiesSupported);
    printf("globalL1CacheSupported = %d\n", device_prop.globalL1CacheSupported);
    printf("localL1CacheSupported = %d\n", device_prop.localL1CacheSupported);
    printf("sharedMemPerMultiprocessor = %zu\n",
           device_prop.sharedMemPerMultiprocessor);
    printf("regsPerMultiprocessor = %d\n", device_prop.regsPerMultiprocessor);
    printf("managedMemory = %d\n", device_prop.managedMemory);
    printf("isMultiGpuBoard = %d\n", device_prop.isMultiGpuBoard);
    printf("multiGpuBoardGroupID = %d\n", device_prop.multiGpuBoardGroupID);
    printf("singleToDoublePrecisionPerfRatio = %d\n",
           device_prop.singleToDoublePrecisionPerfRatio);
    printf("pageableMemoryAccess = %d\n", device_prop.pageableMemoryAccess);
    printf("concurrentManagedAccess = %d\n",
           device_prop.concurrentManagedAccess);
    printf("computePreemptionSupported = %d\n",
           device_prop.computePreemptionSupported);
    printf("canUseHostPointerForRegisteredMem = %d\n",
           device_prop.canUseHostPointerForRegisteredMem);
    printf("cooperativeLaunch = %d\n", device_prop.cooperativeLaunch);
    printf("cooperativeMultiDeviceLaunch = %d\n",
           device_prop.cooperativeMultiDeviceLaunch);
    printf("pageableMemoryAccessUsesHostPageTables = %d\n",
           device_prop.pageableMemoryAccessUsesHostPageTables);
    printf("directManagedMemAccessFromHost = %d\n",
           device_prop.directManagedMemAccessFromHost);
  }
}
int main() {
  {
    int cuda_dev_number;
    check_cuda_errors(cudaGetDeviceCount(&cuda_dev_number));
    printf("cuda_dev_number=%d\n", cuda_dev_number);
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