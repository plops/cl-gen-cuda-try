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
    cudaFreeHost(a);
    cudaFree(d_a);
    cudaFreeHost(b);
    cudaFree(d_b);
    cudaFreeHost(c);
    cudaFree(d_c);
    return 0;
  }
}