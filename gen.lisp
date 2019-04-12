(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-cpp-generator))
(in-package :cl-cpp-generator)
(defmacro e (&body body)
  `(statements (<< "std::cout" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))



;; ssh -p 1235 localhost -L 5900:localhost:5900 -L 2221:10.1.10.3:22
;; https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm
;; https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/technologies/turing-architecture/NVIDIA-Turing-Architecture-Whitepaper.pdf
;; L2 cache/gpu      =  5.6MB
;; register file/gpu = 17.4MB = 72 (68?) * 256kB
;; L1 6.9MB = 72*96kB (64l1,32shm) or (32l1,64shm) (shared between 4 processing blocks)
;; Compute workloads can divide the 96 KB into 32 KB
;; shared memory and 64 KB L1 cache, or 64 KB shared
;; memory and 32 KB L1 cache.

;; https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf
;; tensor core
;; mixed precision fp32/fp16 2d fft https://sc18.supercomputing.org/proceedings/src_poster/poster_files/spost111s2-file2.pdf
;; tensor Core Programmability,Performance & Precision (2018) https://arxiv.org/pdf/1803.04014

;; W. Linderman, J. Corner, and S. Tucker. 2006. Real-time wide swath synthetic aperture radar image formation using embedded HPC. InHPCMP Users GroupConferences, 2006. IEEE, 244–251. 10.1109/hpcmp-ugc.2006.68

;; real-time formation of 37 km wide strips of imagery with
;; <1m resolution.  Twenty-four dual Xeon nodes costing less
;; than $100K are capable of sustaining the real-time
;; throughput of 100 GFLOPS and continuously produce imagery
;; at a rate of 3.43 km^2/second.



#+nil
(defun rev (x nn)
  (let ((n (floor (log nn 2)))
      (res 0))
  (dotimes (i n)
    (setf (ldb (byte 1 i) res) (ldb (byte 1 (- n 1 i)) x)))
  res))

#+nil
(code `(with-compilation-unit
	   (include <iostream>)
	 (include <iomanip>)		 
	 (include <cmath>)
	 (include <algorithm>)
	 (include <array>)
	 (include <complex>)
	 (include <sys/time.h>) ;; gettimeofday

	 (with-compilation-unit
	     (enum Constants (M_MAG_N ,n))
		 
	   (decl (
			  
		  (m_fft_in :type "std::array<std::complex<float>,M_MAG_N>"  :init (list (list ,@ (loop for i below n collect 0.0))))
		  (m_fft_out :type "std::array<std::complex<float>,M_MAG_N>"  :init (list (list ,@ (loop for i below n collect 0.0))))
		  (m_fft_out2 :type "std::array<std::complex<float>,M_MAG_N>"  :init (list (list ,@ (loop for i below n collect 0.0))))
		  (m_fft_out_mag :type "std::array<float,M_MAG_N>" :init (list (list ,@ (loop for i below n collect 0.0))))
		  )))
	 (function (current_time () "static inline uint64_t")
                                         
                   (let ((tv :type "struct timeval"))
                     (funcall gettimeofday &tv nullptr)
                     (return (+ (* tv.tv_sec 1000000)
                                tv.tv_usec))))

	 (function (ft ((in :type "const std::array<std::complex<float>, N > &" )
			(out :type "std::array<std::complex<float>, N > &" ))
		       "template<std::size_t N> void"
		       )
		   (dotimes (k N)
		     (setf (aref out k) 0))
		   (dotimes (k N)
		     (dotimes (n N)
		       (+= (aref out k) (* (funcall "std::exp"
						    (funcall "std::complex<float>"
							     0s0
							     (/ (* M_PI -2 k n)
								N)))
					   (aref in n))))))
	 (function  (bit_reverse_copy ((in :type "const std::array<std::complex<float>, N > &")
				       (out :type "std::array<std::complex<float>, N > &"))
				      "template<std::size_t N > void")
		    (setf ,@(loop for i below n appending
				 `((aref out ,(rev i n)) (aref in ,i)))))
	 (function
	  (fft ((in :type "const std::array<std::complex<float>, N > &")
		(out :type "std::array<std::complex<float>, N > &"))
	       "template<std::size_t N > void")
	  (funcall bit_reverse_copy in out)
	  ,@(loop for s from 1 upto (floor (log n 2)) appending
		 (let ((m (expt 2 s)))
		   `((let ((w_m :type "const auto" :init (funcall "std::complex<float>"
								  ,(coerce (cos (/ (* pi -2) m)) 'single-float)
								  ,(coerce (sin (/ (* pi -2) m)) 'single-float))))
		       (for ((k 0) (< k N) (+= k ,m))
			    (let ((w :type "std::complex<float>" :ctor 1))
			      (dotimes (j ,(/ m 2))
				(let ((t :ctor (* w (aref out (+ k j ,(/ m 2)))))
				      (u :ctor (aref out (+ k j)))
				      )
				  (setf (aref out (+ k j)) (+ u t)
					(aref out (+ k j ,(/ m 2))) (- u t)
					w (* w w_m)))))))))
		 ))
	      
	       
	 (function (main () int)

		   (statements
		    (dotimes (i M_MAG_N)
		      (setf (aref m_fft_in i) 0.0
			    (aref m_fft_out i) 0.0
			    (aref m_fft_out_mag i) 0.0))
		    (setf (aref m_fft_in 1) 1.0)
		    (macroexpand (benchmark
				  (dotimes (i 10)
				    (funcall ft m_fft_in m_fft_out)))))

		   (statements
		    (dotimes (i M_MAG_N)
		      (setf (aref m_fft_in i) 0.0
			    (aref m_fft_out2 i) 0.0
			    (aref m_fft_out_mag i) 0.0))
		    (setf (aref m_fft_in 1) 1.0)
		    (macroexpand (benchmark
				  (dotimes (i 10)
				    (funcall fft m_fft_in m_fft_out2)))))
			 
		   #+nil (dotimes  (i M_MAG_N)
			   (setf (aref m_fft_out_mag i) (funcall "std::abs" (aref m_fft_out i))))
			 
		   (dotimes (i M_MAG_N)
		     (macroexpand (e (funcall "std::setw" 6) i (funcall "std::setw" 30) (aref m_fft_out i) (funcall "std::setw" 30) (aref m_fft_out2 i)))))))


(defparameter *device-attribute*
  `((cudaDevAttrMaxThreadsPerBlock "Maximum number of threads per block")
    (cudaDevAttrMaxBlockDimX "Maximum x-dimension of a block")
    (cudaDevAttrMaxBlockDimY "Maximum y-dimension of a block")
    (cudaDevAttrMaxBlockDimZ "Maximum z-dimension of a block")
    (cudaDevAttrMaxGridDimX "Maximum x-dimension of a grid")
    (cudaDevAttrMaxGridDimY "Maximum y-dimension of a grid")
    (cudaDevAttrMaxGridDimZ "Maximum z-dimension of a grid")
    (cudaDevAttrMaxSharedMemoryPerBlock "Maximum amount of shared memoryavailable to a thread block in bytes")
    (cudaDevAttrTotalConstantMemory "Memory available on device for __constant__variables in a CUDA C kernel in bytes")
    (cudaDevAttrWarpSize "Warp size in threads")
    (cudaDevAttrMaxPitch "Maximum pitch in bytes allowed by the memory copyfunctions that involve memory regions allocated through cudaMallocPitch()")
    (cudaDevAttrMaxTexture1DWidth "Maximum 1D texture width")
    (cudaDevAttrMaxTexture1DLinearWidth "Maximum width for a 1D texture boundto linear memory")
    (cudaDevAttrMaxTexture1DMipmappedWidth "Maximum mipmapped 1D texturewidth")
    (cudaDevAttrMaxTexture2DWidth "Maximum 2D texture width")
    (cudaDevAttrMaxTexture2DHeight "Maximum 2D texture height")
    (cudaDevAttrMaxTexture2DLinearWidth "Maximum width for a 2D texture boundto linear memory")
    (cudaDevAttrMaxTexture2DLinearHeight "Maximum height for a 2D texture boundto linear memory")
    (cudaDevAttrMaxTexture2DLinearPitch "Maximum pitch in bytes for a 2D texturebound to linear memory")
    (cudaDevAttrMaxTexture2DMipmappedWidth "Maximum mipmapped 2D texturewidth")
    (cudaDevAttrMaxTexture2DMipmappedHeight "Maximum mipmapped 2D textureheight")
    (cudaDevAttrMaxTexture3DWidth "Maximum 3D texture width")
    (cudaDevAttrMaxTexture3DHeight "Maximum 3D texture height")
    (cudaDevAttrMaxTexture3DDepth "Maximum 3D texture depth")
    (cudaDevAttrMaxTexture3DWidthAlt "Alternate maximum 3D texture width, 0 if noalternate maximum 3D texture size is supported")
    (cudaDevAttrMaxTexture3DHeightAlt "Alternate maximum 3D texture height, 0 ifno alternate maximum 3D texture size is supported")
    (cudaDevAttrMaxTexture3DDepthAlt "Alternate maximum 3D texture depth, 0 if noalternate maximum 3D texture size is supported")
    (cudaDevAttrMaxTextureCubemapWidth "Maximum cubemap texture width orheight")
    (cudaDevAttrMaxTexture1DLayeredWidth "Maximum 1D layered texture width")
    (cudaDevAttrMaxTexture1DLayeredLayers "Maximum layers in a 1D layeredtexture")
    (cudaDevAttrMaxTexture2DLayeredWidth "Maximum 2D layered texture width")
    (cudaDevAttrMaxTexture2DLayeredHeight "Maximum 2D layered texture height")
    (cudaDevAttrMaxTexture2DLayeredLayers "Maximum layers in a 2D layeredtexture")
    (cudaDevAttrMaxTextureCubemapLayeredWidth "Maximum cubemap layeredtexture width or height")
    (cudaDevAttrMaxTextureCubemapLayeredLayers "Maximum layers in a cubemaplayered texture")
    (cudaDevAttrMaxSurface1DWidth "Maximum 1D surface width")
    (cudaDevAttrMaxSurface2DWidth "Maximum 2D surface width")
    (cudaDevAttrMaxSurface2DHeight "Maximum 2D surface height")
    (cudaDevAttrMaxSurface3DWidth "Maximum 3D surface width")
    (cudaDevAttrMaxSurface3DHeight "Maximum 3D surface height")
    (cudaDevAttrMaxSurface3DDepth "Maximum 3D surface depth")
    (cudaDevAttrMaxSurface1DLayeredWidth "Maximum 1D layered surface width")
    (cudaDevAttrMaxSurface1DLayeredLayers "Maximum layers in a 1D layeredsurface")
    (cudaDevAttrMaxSurface2DLayeredWidth "Maximum 2D layered surface width")
    (cudaDevAttrMaxSurface2DLayeredHeight "Maximum 2D layered surface height")
    (cudaDevAttrMaxSurface2DLayeredLayers "Maximum layers in a 2D layeredsurface")
    (cudaDevAttrMaxSurfaceCubemapWidth "Maximum cubemap surface width")
    (cudaDevAttrMaxSurfaceCubemapLayeredWidth "Maximum cubemap layeredsurface width")
    (cudaDevAttrMaxSurfaceCubemapLayeredLayers "Maximum layers in a cubemaplayered surface")
    (cudaDevAttrMaxRegistersPerBlock "Maximum number of 32-bit registers availableto a thread block")
    (cudaDevAttrClockRate "Peak clock frequency in kilohertz")
    (cudaDevAttrTextureAlignment "Alignment requirement texture base addressesaligned to textureAlign bytes do not need an offset applied to texture fetches")
    (cudaDevAttrTexturePitchAlignment "Pitch alignment requirement for 2D texturereferences bound to pitched memory")
    (cudaDevAttrGpuOverlap "1 if the device can concurrently copy memory betweenhost and device while executing a kernel, or 0 if not")
    (cudaDevAttrMultiProcessorCount "Number of multiprocessors on the device")
    (cudaDevAttrKernelExecTimeout "1 if there is a run time limit for kernels executedon the device, or 0 if not")
    (cudaDevAttrIntegrated "1 if the device is integrated with the memory subsystem, or0 if not")
    (cudaDevAttrCanMapHostMemory "1 if the device can map host memory into theCUDA address space, or 0 if not")
    (cudaDevAttrComputeMode "Compute mode is the compute mode that the device iscurrently in. Available modes are as follows")
    (cudaDevAttrConcurrentKernels "1 if the device supports executing multiple kernelswithin the same context simultaneously, or 0 if not. It is not guaranteed that multipkernels will be resident on the device concurrently so this feature should not berelied upon for correctness")
    (cudaDevAttrEccEnabled "1 if error correction is enabled on the device, 0 if errorcorrection is disabled or not supported by the device")
    (cudaDevAttrPciBusId "PCI bus identifier of the device")
    (cudaDevAttrPciDeviceId "PCI device (also known as slot) identifier of the device")
    (cudaDevAttrTccDriver "1 if the device is using a TCC driver. TCC is only availableon Tesla hardware running Windows Vista or later")
    (cudaDevAttrMemoryClockRate "Peak memory clock frequency in kilohertz")
    (cudaDevAttrGlobalMemoryBusWidth "Global memory bus width in bits")
    (cudaDevAttrL2CacheSize "Size of L2 cache in bytes. 0 if the device doesn't have L2cache")
    (cudaDevAttrMaxThreadsPerMultiProcessor "Maximum resident threads permultiprocessor")
    (cudaDevAttrUnifiedAddressing "1 if the device shares a unified address space withthe host, or 0 if not")
    (cudaDevAttrComputeCapabilityMajor "Major compute capability version number")
    (cudaDevAttrComputeCapabilityMinor "Minor compute capability version number")
    (cudaDevAttrStreamPrioritiesSupported "1 if the device supports stream priorities, or0 if not")
    (cudaDevAttrGlobalL1CacheSupported "1 if device supports caching globals in L1cache, 0 if not")
    (cudaDevAttrLocalL1CacheSupported "1 if device supports caching locals in L1cache, 0 if not")
    (cudaDevAttrMaxSharedMemoryPerMultiprocessor "Maximum amount of sharedmemory available to a multiprocessor in bytes this amount is shared by all threadblocks simultaneously resident on a multiprocessor")
    (cudaDevAttrMaxRegistersPerMultiprocessor "Maximum number of 32-bitregisters available to a multiprocessor this number is shared by all thread blockssimultaneously resident on a multiprocessor")
    (cudaDevAttrManagedMemory "1 if device supports allocating managed memory, 0 if not")
    (cudaDevAttrIsMultiGpuBoard "1 if device is on a multi-GPU board, 0 if not")
    (cudaDevAttrMultiGpuBoardGroupID "Unique identifier for a group of devices onthe same multi-GPU board")
    (cudaDevAttrHostNativeAtomicSupported "1 if the link between the device and thehost supports native atomic operations")
    (cudaDevAttrSingleToDoublePrecisionPerfRatio "Ratio of single precisionperformance (in floating-point operations per second) to double precision performance")
    (cudaDevAttrPageableMemoryAccess "1 if the device supports coherently accessingpageable memory without calling cudaHostRegister on it, and 0 otherwise.")
    (cudaDevAttrConcurrentManagedAccess "1 if the device can coherently accessmanaged memory concurrently with the CPU, and 0 otherwise.")
    (cudaDevAttrComputePreemptionSupported "1 if the device supports ComputePreemption, 0 if not.")
    (cudaDevAttrCanUseHostPointerForRegisteredMem "1 if the device can access hostregistered memory at the same virtual address as the CPU, and 0 otherwise.")
    (cudaDevAttrCooperativeLaunch "1 if the device supports launching cooperativekernels via cudaLaunchCooperativeKernel, and 0 otherwise.")
    (cudaDevAttrCooperativeMultiDeviceLaunch "1 if the device supports launchingcooperative kernels via cudaLaunchCooperativeKernelMultiDevice, and 0otherwise.")
    (cudaDevAttrCanFlushRemoteWrites "1 if the device supports flushing ofoutstanding remote writes, and 0 otherwise.")
    (cudaDevAttrHostRegisterSupported  "1 if the device supports host memoryregistration via cudaHostRegister, and 0 otherwise.")
    (cudaDevAttrPageableMemoryAccessUsesHostPageTables "1 if the device accessespageable memory via the host's page tables, and 0 otherwise.")
    (cudaDevAttrDirectManagedMemAccessFromHost "1 if the host can directly accessmanaged memory on the device without migration, and 0 otherwise.")
    (cudaDevAttrMaxSharedMemoryPerBlockOptin "Maximum per blockshared memory size on the device. This value can be opted into when usingcudaFuncSetAttribute")) )


(progn
  (defparameter *main-cpp-filename*
    (merge-pathnames "stage/cl-gen-cuda-try/source/cuda_try"
		     (user-homedir-pathname)))
  (let* (;(n 32)
	 (code
	  `(with-compilation-unit
	       (include <stdio.h>)
	     
	     (raw "// https://www.youtube.com/watch?v=Ed_h2km0liI CUDACast #2 - Your First CUDA C Program")
	     (raw "// https://github.com/NVIDIA-developer-blog/cudacasts/blob/master/ep2-first-cuda-c-program/kernel.cu")
	     (function ("vector_add" ((a :type "int*")
				      (b :type "int*")
				      (c :type "int*")
				      (n :type "int"))
				     "__global__ void")
		       
		       (let ((i :type int :init threadIdx.x))
			 (if (< i n)
			     (statements
			      (setf (aref c i) (+ (aref a i)
						  (aref b i)))))))
	     (enum () (N 1024))
	     (function ("main" ()
			       "int")
		       (let ((cuda_dev :type int))
			 (funcall cudaChooseDevice &cuda_dev NULL)
			 (raw "// read device attributes")
			 (let ((val :type int))
			   ,@(loop for (attr text) in *device-attribute* collect
				  `(statements
				    (funcall cudaDeviceGetAttribute &val ,attr cuda_dev)
				    (funcall printf (string ,(format nil "~a=%d (~a)\\n" attr text)) val)))))
		       #+nil(let ((device_prop :type cudaDeviceProp))
			 (funcall cudaGetDeviceProperties &device_prop cuda_dev)
			 )
		       
		       (let (,@(loop for e in '(a b c) collect
				    `(,e :type int* :init (funcall "static_cast<int*>"
								   (funcall malloc (* N (funcall sizeof int))))))
			     ,@(loop for e in '(a b c) collect
				    `(,(format nil "d_~a" e) :type int*))
			       )
			 ,@(loop for e in '(a b c) collect
				`(funcall cudaMalloc ,(format nil "&d_~a" e) (* N (funcall sizeof int))))
			 (dotimes (i N)
			   (setf (aref a i) i
				 (aref b i) i
				 (aref c i) 0))
			 ,@(loop for e in '(a b c) collect
				`(funcall cudaMemcpy ,(format nil "d_~a" e) ,e (* N (funcall sizeof int))
					  cudaMemcpyHostToDevice))
			 (funcall "vector_add<<<1,N>>>" d_a d_b d_c N)
			 (funcall cudaMemcpy c d_c (* N (funcall sizeof int))
				  cudaMemcpyDeviceToHost)
			 ,@(loop for e in '(a b c) collect
				`(statements
				  (funcall free ,e)
				  (funcall cudaFree ,(format nil "d_~a" e))))
			 (return 0))))))
    (write-source *main-cpp-filename* "cu" code)
    (sb-ext:run-program "/usr/bin/scp" `("-C" ,(format nil "~a.cu" *main-cpp-filename*) "-l" "root" "vast:./"))))


