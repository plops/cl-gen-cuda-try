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


;; exact representation of floating point constants
;; [1] http://clhs.lisp.se/Body/f_dec_fl.htm    decode-float float => significand, exponent, sign
;; [2] https://www.exploringbinary.com/hexadecimal-floating-point-constants/ examples of c hex notation 
;; [3] http://www.open-std.org/jtc1/sc22/wg14/www/docs/n1256.pdf page 57, parsing

;; 0x1.999999999999ap-4 is an example of a normalized,
;; double-precision hexadecimal floating-point constant; it represents
;; the double-precision floating-point number nearest to the decimal
;; number 0.1. The constant is made up of four parts:

    ;; The prefix ‘0x’, which shows it’s a hexadecimal constant.
    ;; A one hex digit integer part ‘1’, which represents the leading 1 bit of a normalized binary fraction.
    ;; A thirteen hex digit fractional part ‘.999999999999a’, which represents the remaining 52 significant bits of the normalized binary fraction.
    ;; The suffix ‘p-4’, which represents the power of two, written in decimal: 2-4.


;; 0x1.99999ap-4 is the single-precision constant representing
;; 0.1. Single-precision values don’t map as neatly to hexadecimal
;; constants as double-precision values do; single-precision is 24
;; bits, but a normalized hexadecimal constant shows 25 bits. This is
;; not a problem, however; the last hex digit will always have a
;; binary equivalent ending in 0.

;; translation-time conversion of floating constants should match the
;; execution-timeconversion of character strings by library functions,
;; such as strtod


(sb-posix:strtod "0x1.999999999999ap-4") ;; => 0.1d0, 20
(integer-decode-float (sb-posix:strtod "0x1.999999999999ap-4")) ;; => 7205759403792794, -56, 1
(format nil "~{~x~^ ~}" (multiple-value-list (integer-decode-float (sb-posix:strtod "0x1.999999999999ap-4")))) ;; => "1999999999999A -38 1"


(sb-posix:strtod "0x1.99999ap-4") 


(defun strtof/base-string (chars offset)
  (declare (simple-base-string chars))
  ;; On x86, dx arrays are quicker to make than aliens.
  (sb-int:dx-let ((end (make-array 1 :element-type 'sb-ext:word)))
    (sb-sys:with-pinned-objects (chars)
      (let* ((base (sb-sys:sap+ (sb-sys:vector-sap chars) offset))
	     (answer
	      (handler-case
		  (sb-alien:alien-funcall
		   (sb-alien:extern-alien "strtof" (function sb-alien:float
							     sb-alien:system-area-pointer
							     sb-alien:system-area-pointer))
		   base
		   (sb-sys:vector-sap end))
		(floating-point-overflow () nil))))
	(values answer
		(if answer
		    (the sb-int:index
			 (- (aref end 0) (sb-sys:sap-int base)))))))))


(strtof/base-string (coerce "0x1.99999ap-4" 'simple-base-string) 0) ;; => 0.1, 13

(type-of (strtof/base-string (coerce "0x1.99999ap-4" 'simple-base-string) 0)) ;; => single-float

(integer-decode-float (strtof/base-string (coerce "0x1.99999ap-4" 'simple-base-string) 0)) ;; => 13421773, -27, 1
(integer-decode-float (strtof/base-string (coerce "0x1.99999ap-150" 'simple-base-string) 0))

;; transfer of exponent (seems to be decimal in c)
;; c    lisp
;; -151 0
;; -150 -172
;; -126 -149
;; -4 -27
;; -3 -26
;; 127 104
;; 128 fail
;; 




(multiple-value-bind (a b c) (integer-decode-float (strtof/base-string (coerce "0x1.99999ap1" 'simple-base-string) 0))
  (let ((significand (ash a 1)))
    (format nil "0x~x.~xp~d"
	    (ldb (byte 4 (* 6 4)) significand)
	    (ldb (byte (* 6 4) 0) significand)
	   (+ 23 b))))


(defun single-float-to-c-hex-string (f)
  (declare (type (single-float 0) f))
  (multiple-value-bind (a b c) (integer-decode-float f)
  (let ((significand (ash a 1)))
    (format nil "0x~x.~xp~d"
	    (ldb (byte 4 (* 6 4)) significand)
	    (ldb (byte (* 6 4) 0) significand)
	   (+ 23 b)))))


(single-float-to-c-hex-string .1s0) ;; => "0x1.99999Ap-4"

(single-float-to-c-hex-string (strtof/base-string (coerce "0x1.99999ap-4" 'simple-base-string) 0))


(format nil "~{~x~^ ~}" (multiple-value-list (integer-decode-float (strtof/base-string (coerce "0x1.99999ap-4" 'simple-base-string) 0))))  ;; => "CCCCCD -1B 1"


(format nil "~b" #xCCCCCD)  ;; "110011001100110011001101"
(format nil "~b" #x199999a) ;; "1100110011001100110011010"

(ash #xCCCCCD 1) ;; => 26843546 (25 bits, #x199999A, #o146314632, #b1100110011001100110011010)


(decode-float 0.1s0)
(decode-float 0.1d0)
(integer-decode-float 0.1s0)
(integer-decode-float 0.1d0)


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
				   (con :type "const alignas(64) vsf" :init (list ,@(loop for i below simd-length
											 collect
											   (* 1s0 i))))
				   ,@(let ((args-seen () #+nil (list 0 1/4 -1/4 3/4 1/2)))
				       (loop for k2 below n2 appending ;; column
					    (loop for n2_ below n2
					       when (not (member (twiddle-arg n2_ k2 n2) args-seen))
					       appending
						 (progn
						   (push (twiddle-arg n2_ k2 n2) args-seen)
						   (list
						    `(,(format nil "w~a_re" (twiddle-arg-name n2_ k2 n2)) :type "const float"
						       :init ,(realpart (flush-z (exp (complex 0s0 (* -2 (/ pi n2) n2_ k2))))))
						    `(,(format nil "w~a_im" (twiddle-arg-name n2_ k2 n2)) :type "const float"
						       :init ,(imagpart (flush-z (exp (complex 0s0 (* -2 (/ pi n2) n2_ k2))))))))))))
			       ,@(loop for k2 below n2 appending 
				      (loop for n1_ below (/ n1 simd-length) collect
					   `(setf ,(row-major 'x1_re n1_ k2)
						  (+ 
						   ,@(loop for n2_ below n2 collect
							  `(* con 
							     ,(row-major 're_in n1_ n2_))
							  #+nil
							  (twiddle-mul (row-major 're_in n1_ n2_)
								       n2_ k2 n2))))))
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
			     (funcall simd_driver)
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


