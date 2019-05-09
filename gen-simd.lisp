(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-cpp-generator))
(in-package :cl-cpp-generator)


(setf *features* (union *features* '(:memset)))
;(setf *features* (set-difference *features* '(:memset)))

(defmacro e (&body body)
  `(statements (<< "std::cout" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))



(defparameter *facts*
  `((10 "to make use of cache read sequentially write random (from limited range)")
    (20 "is single cycle sinf, cosf good enough? https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html")
    (30 "compute twiddle factors using addition theorem exp(x+y)=exp(x)*exp(y)")
    (40 "only store twiddle factors that are necessary")
    (50 "radix 4 and 2 are preferred as they don't require floating point multiplication in lower stages")
    (60 "will __builtin_prefetch help with strided memory access and make transposition unneccessary? https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html")
    (70 "in order to simplify implementing simd with complex numbers, store real and imaginary parts in separate arrays")))


(let ((r 4))
  (* 2 r (log 16 r)))

(defun rev (x nn)
  (let ((n (floor (log nn 2)))
	(res 0))
    (dotimes (i n)
      (setf (ldb (byte 1 i) res) (ldb (byte 1 (- n 1 i)) x)))
    res))


(progn
  (defun flush (a)
  (if (< (abs a) 1e-15)
      0s0
      a))
  (defun flush-z (z)
    (let ((a (realpart z))
	  (b (imagpart z)))
      (complex (flush a) (flush b))))
  
  (defun twiddle-arg (j k n)
    "Twiddle factors are named by their angle in the unit turn turn https://en.wikipedia.org/wiki/Turn_(geometry). Storing it as a rational number doesn't loose precision."
    (- (mod (+ (/ 1 n) (/ (* -1 j k)
			   n))
	    1)
       (/ 1 n)))
  (defun twiddle-arg-name (j k n)
    (let ((arg (twiddle-arg j k n)))
      (format nil "~a~a~a_~a" n
	      (if (< arg 0)
		  "m"
		  "p")
	      (abs (numerator arg))
	      (denominator arg))))

  (defun twiddle-mul (e
		       j k n)
    "compute product of a complex number and a twiddle factor. express without multiplication if possible."
    (case (twiddle-arg j k n)	;if (eq 0 (* j2 k))
      (0 e)
      (1/2 `(* -1 ,e))
      (1/4 `(* (funcall ;;__builtin_complex ;;
		CMPLXF
		(* -1 (funcall cimagf ,e))
		(funcall crealf ,e))))
      (-1/4 `(* (funcall ;; __builtin_complex ;;
		 CMPLXF
		 (funcall cimagf ,e)
		 (* -1 (funcall crealf ,e)))))
      (3/4 `(* (funcall ;; __builtin_complex ;;
		 CMPLXF
		 (funcall cimagf ,e)
		 (* -1 (funcall crealf ,e)))))
      (t `(* ,e
	     ,(format nil "w~a" (twiddle-arg-name j k n))))))
  
  (defparameter *main-cpp-filename*
    (merge-pathnames "stage/cl-gen-cuda-try/source/simd_try"
		     (user-homedir-pathname)))
  (let* ((n1 4)
	 (n2 4)
	 (n (* n1 n2))
	 (code
	  `(with-compilation-unit

	     ;; https://news.ycombinator.com/item?id=13147890
	       (raw "//gcc -std=c11 -Ofast -flto -ffast-math -march=skylake -msse2  -ftree-vectorize -mfma -mavx2")
	     (raw " ")
	     ;; https://dendibakh.github.io/blog/2017/10/30/Compiler-optimization-report
	     (raw "//clang -std=c11 -Ofast -flto -ffast-math -march=skylake -msse2 -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize")
	     (raw " ")
	     (raw "//icc -std=c11 -O2 -D NOFUNCCALL -qopt-report=1 -qopt-report-phase=vec -guide-vec -parallel")
	     ;; icc with -O3 does cache blocking and prefetching https://www.youtube.com/watch?v=nvPYYE0OWVA
	     ;; don't link with -lm explicitly icc
	     (raw " ")
	     
	     (include <stdio.h>)
	     (include <stdlib.h>)
	     (include <string.h>)
	     (include <complex.h>)
	     (include <stdalign.h>)
	     (include <math.h>)
					;(raw "#typedef scomplex float complex")

	     (raw "#ifndef CMPLXF")
	     (raw "#define CMPLXF(real, imag) ((real) + _Complex_I * (imag))")
	     (raw "#endif")
	     
	     ,(let* ((simd-length 16)
		     (simd-name (format nil "vsf" ; simd-length
					))
		     (n1 simd-length)
		     (n2 7)
		     (n (* n1 n2))
		     (fft (format nil "simd_~a_fft_~a_~a_~a" simd-length n n1 n2)))
		(flet ((row-major (a x y)
			 `(aref ,a ,(+ (* 1 x) (* n1 y))))
		       (col-major (a x y)
			 `(aref ,a ,(+ (* n2 x) (* 1 y)))))
		 `(statements
		   (raw ,(format nil "typedef float ~a __attribute__ ((vector_size (~a)));"
				 simd-name (* 4 simd-length)))
		   
		   (function (,fft (,@(loop for e in '(re_in im_in re_out im_out) collect
					   `(,e :type "vsf* __restrict__")))
				   float)
			     ,@(loop for e in '(re_in im_in re_out im_out) collect
				    `(setf ,e (funcall __builtin_assume_aligned ,e 64)))
			     (let (((aref x1_re ,(* (/ n1 simd-length) n2)) :type "static alignas(64) vsf")
				   ((aref x1_im ,(* (/ n1 simd-length) n2)) :type "static alignas(64) vsf")
				   #+nil (con :type "const alignas(64) vsf" :init (list ,@(loop for i below simd-length
											 collect
											   (* 1s0 i))))
				   ,@(labels ((c-hex-float-name (v)
						(declare (type (single-float 0) v))
						(format nil "~{~a~^_~}"
							(mapcar (lambda (x) (if (< x 0)
										(format nil "m~a" (abs x))
										x))
								(multiple-value-list (integer-decode-float (abs v))))))
					      (c-hex-float-def (v)
						`(,(format nil "w~a /* ~a */"
							   (c-hex-float-name v)
							   v)
						   :type "const float"
						   :init (hex ,v))))
				       (mapcar #'c-hex-float-def
					       (sort
						(remove-duplicates
						 (loop for k2 below n2 appending
						      (loop for n2_ below n2 appending
							   (let ((u (coerce (abs (realpart (flush-z (exp (complex 0s0 (* -2 (/ pi n2) n2_ k2))))))
									    'single-float
									    ))
								 (v (coerce (abs (imagpart (flush-z (exp (complex 0s0 (* -2 (/ pi n2) n2_ k2))))))
									    'single-float
									    )))
							     `(,u ,v)))))
						#'<))))
			       ,@(loop for k2 below n2 appending 
				      (loop for n1_ below (/ n1 simd-length) collect
					   `(let ((coef_re :type "const alignas(64) vsf"
							   :init (list ,@(loop for i below simd-length
									    collect
									      (let ((v (coerce (realpart (flush-z (exp (complex 0s0 (* -2 (/ pi n2) i k2)))))
											       'single-float)))
										`(* ,(floor (signum v))
										    ,(format nil "w~{~a~^_~}" (mapcar (lambda (x) (if (< x 0)
																      (format nil "m~a" (abs x))
																      x))
														      (multiple-value-list (integer-decode-float (abs v))))
											     ))))))
						  (coef_im :type "const alignas(64) vsf"
							   :init (list ,@(loop for i below simd-length
									    collect
									      (let* ((v (coerce (imagpart (flush-z (exp (complex 0s0 (* -2 (/ pi n2) i k2)))))
												  'single-float))
										     (name (format nil "w~{~a~^_~}" (mapcar (lambda (x) (if (< x 0)
																      (format nil "m~a" (abs x))
																      x))
															    (multiple-value-list (integer-decode-float (abs  v))))
											     )))
										(if (< (floor (signum v)) 0)
										    `(* -1 ,name)
										    name)))))
						  )
					      (setf ,(row-major 'x1_re n1_ k2)
						    (+ 
						     ,@(loop for n2_ below n2 collect
							    `(- (* coef_re ,(row-major 're_in n1_ n2_))
								(* coef_im ,(row-major 'im_in n1_ n2_))))))
					      (setf ,(row-major 'x1_im n1_ k2)
						    (+ 
						     ,@(loop for n2_ below n2 collect
							    `(+ (* coef_im ,(row-major 're_in n1_ n2_))
								(* coef_re ,(row-major 'im_in n1_ n2_)))))))))
			       (funcall memcpy re_out x1_re (funcall sizeof x1_re))
			       #+nil(dotimes (j ,n2)
				 (dotimes (i ,(/ n1 simd-length))
				   (dotimes (k ,simd-length)
				     (funcall printf (string "%f\\n") (aref x1_re (+ i (* j ,(/ n1 simd-length)))
									    k)))))
			       (return (aref x1_re 0 0))))
		   (function (simd_driver ()
					  void)
			     (let (,@ (loop for e in '(in_re in_im out_re out_im)
					 collect
					   `((aref ,e ,(* (/ n1 simd-length) n2)) :type "static vsf"))
				   )
			       (funcall ,fft in_re in_im out_re out_im)
			       (dotimes (i ,simd-length)
				 (funcall printf (string "%f\\n") (aref out_re 0 i)))))
		   (function ("main" ()
				     int)
			     (let ((a :type float :init (hex .1s0)))
			      (funcall simd_driver))
			     (return 0))))))))
    (write-source *main-cpp-filename* "c" code)
    ;(uiop:run-program "clang -Wextra -Wall -march=native -std=c11 -Ofast -ffast-math -march=native -msse2  source/cpu_try.c -g -o source/cpu_try_clang -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -lm 2>&1 > source/cpu_try_clang.out")
					;(uiop:run-program "clang -Wextra -Wall -march=native -std=c11 -Ofast -ffast-math -march=native -msse2  source/cpu_try.c -S -o source/cpu_try_clang.s -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -lm")
    ;; -Ofast -ffast-math
    (uiop:run-program "gcc -Wall -Wextra -std=c11 -g -O3 -march=native -mfma -ffast-math source/simd_try.c -o source/simd_try_gcc -lm  2>&1 > source/simd_try_gcc.out")
    (uiop:run-program "gcc -march=native -std=c11 -O3 -march=native -mfma -ffast-math -S source/simd_try.c -o source/simd_try_gcc.s")
    ))




;; valgrind --tool=cachegrind source/cpu_try_gcc
;; cg_annotate cachegrind.out.3788


