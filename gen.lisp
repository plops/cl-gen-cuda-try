(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-cpp-generator))
(in-package :cl-cpp-generator)
(defmacro e (&body body)
  `(statements (<< "std::cout" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))



;; ssh -p 1235 localhost -L 5900:localhost:5900 -L 2221:10.1.10.3:22

(defun rev (x nn)
  (let ((n (floor (log nn 2)))
      (res 0))
  (dotimes (i n)
    (setf (ldb (byte 1 i) res) (ldb (byte 1 (- n 1 i)) x)))
  res))


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
	       ;; https://arxiv.org/pdf/1803.04014
	       
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

(progn
  (defparameter *main-cpp-filename*
    (merge-pathnames "stage/cl-gen-cuda-try/source/cuda_try"
		     (user-homedir-pathname)))
  (let* ((n 32)
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


