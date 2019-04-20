(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-cpp-generator))
(in-package :cl-cpp-generator)

(defmacro e (&body body)
  `(statements (<< "std::cout" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))



(defparameter *facts*
  `((10 "to make use of cache read sequentially write random (from limited range)")
    (20 "is single cycle sinf, cosf good enough? https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html")
    (30 "compute twiddle factors using addition theorem exp(x+y)=exp(x)*exp(y)")
    (40 "only store twiddle factors that are necessary")
    (50 "radix 4 and 2 are preferred as they don't require floating point multiplication in lower stages")))


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
      (t `(* ,e
	     ,(format nil "w~a" (twiddle-arg-name j k n))))))
  
  (defparameter *main-cpp-filename*
    (merge-pathnames "stage/cl-gen-cuda-try/source/cpu_try"
		     (user-homedir-pathname)))
  (let* ((n1 4)
	 (n2 4)
	 (n (* n1 n2))
	 (code
	  `(with-compilation-unit

	     ;; https://news.ycombinator.com/item?id=13147890
	       (raw "//gcc -std=c99 -Ofast -flto -ffast-math -march=skylake -msse2  -ftree-vectorize -mfma -mavx2")
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


	     
	     (function (fun_slow ((a :type "float complex* __restrict__"))
				 "float complex*")
		       (setf a (funcall __builtin_assume_aligned a 64)) ;; tell compiler that argument ins 64byte aligned
		       (let (((aref y 16) :type "static alignas(64) float complex" :init (list 0.0fi)))
			 #+nil (setf y (funcall aligned_alloc (* 16
							   (funcall sizeof "complex float"))
						64))
			 ;(setf y (funcall __builtin_assume_aligned y 64))
			 (funcall memset y 0 (* 16 (funcall sizeof "complex float")))
			 (dotimes (j 16)
			   (dotimes (k ,n)
			     (setf (aref y j) (+ (aref y j)
						 (* (aref a k)
						    (funcall cexpf (* "1.0fi" ,(* -2 pi (/ n)) j k)))))))
			 (return y)))
	     (function (fun ((x :type "float complex* __restrict__"))
			    "float complex*")
		       (setf x (funcall __builtin_assume_aligned x 64)) ;; tell compiler that argument ins 64byte aligned
		       ,(let ()
			  `(statements
			    (raw "// dft on each row")
			    (raw "// Twiddle factors are named by their angle in the unit turn turn https://en.wikipedia.org/wiki/Turn_(geometry). Storing it as a rational number doesn't loose precision.")
			    (let (((aref s (* ,n1 ,n2)) :type "static float complex" :init (list 0.0fi))
				  ,@(let ((args-seen (list 0 1/4 -1/4 1/2)))
				      (loop for j2 below n2 appending
					   (loop for k below n2 when (not (member
									   (twiddle-arg j2 k n2)
									   args-seen))
					      collect
						(progn
						  (push (twiddle-arg j2 k n2)
							args-seen)
						  `(,(format nil "w~a" (twiddle-arg-name j2 k n2)) :type "const float complex"
						     :init ,(flush-z (exp (complex 0s0 (* -2 (/ pi n2) j2 k))))))))))
			      
			    
			      ,@(loop for j2 below n2 appending
				      (loop for j1 below n1 collect
					   `(setf (aref s ,(+ j1 (* n1 j2)))
						  (+ ,@(loop for k below n2 collect
							    (twiddle-mul `(aref x ,(+ j1 (* k n1)))
									 j2 k n2)
							    )))))

			       (raw "// transpose and elementwise multiplication")
			       (let (((aref z (* ,n1 ,n2)) :type "static float complex" :init (list 0.0fi))
				     ,@(let ((w-seen ()))
					 (loop for j1 below n1 appending
					      (loop for j2 below n2 when (and (/= 0 (* j1 j2))
									      (not 
										  (member (twiddle-arg j1 j2 (* n1 n2))
											  w-seen)))
						 collect
						   (let ((arg (twiddle-arg j1 j2 (* n1 n2))))
						     (push arg
							   w-seen)
						     `(,(format nil "w~a" (twiddle-arg-name j1 j2 (* n1 n2)))
						       :type "const float complex"
						       :init ,(flush-z (exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2))))))))))))
				 ,@(loop for j1 below n1 appending
					(loop for j2 below n2 collect
					     `(setf (aref z ,(+ (* j1 n2) j2))
						    ,(if (eq 0 (* j1 j2))
							 `(aref s ,(+ j1 (* j2 n1)))
							 `(*  (aref s ,(+ j1 (* j2 n1)))
							      ,(format nil "w~a" (twiddle-arg-name j1 j2 (* n1 n2)))
					;,(exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2)))))
							      )))))
				 (raw "// dft on each row")
				 (let (((aref y (* ,n1 ,n2)) :type "static float complex" :init (list 0.0fi))
				       ,@(let ((seen (list 0 1/4 -1/4 1/2)))
					   (loop for j1 below n1 appending
						(loop for j2 below n2 when (and (/= 0 (* j1 j2))
										(not (member (twiddle-arg j1 j2 n1)
											     seen)))
						   collect
						     (let ((arg (twiddle-arg j1 j2 n1)))
						       (push arg seen)
						       `(,(format nil "w~a" (twiddle-arg-name j1 j2 n1))
							  :type "const float complex"
							  :init ,(flush-z (exp (complex 0s0 (* -2 pi j1 j2 (/ n1)))))))))))
				   
				   ,@(loop for j1 below n1 appending
					  (loop for j2 below n2 collect
					       `(setf (aref y ,(+ (* j1 n2) j2))
						      (+ ,@(loop for k below n1 collect
								(twiddle-mul `(aref z ,(+ (* k n2) j2))
									     j1 k n1)
								#+nil
								(if (eq 0 (* j1 k))
								    `(aref z ,(+ (* k n2) j2))
								    `(*
								      ,(format nil "w~a" (twiddle-arg-name j1 k n1))
					;,(exp (complex 0s0 (* -2 pi j1 k (/ n1))))
								      (aref z ,(+ (* k n2) j2)))))))))
				   (return y)))))))	     


	     (decl (((aref global_a (* 4 4)) :type "alignas(64) float complex"
		     :init (list 0.0fi))))
	     (function ("main" ()
			       int)

		       ,@(loop for f in '(0 2 2.5123) collect
			`(statements

			 
			 ,@(loop for i below n collect
				`(setf (aref global_a ,i) ,(cos (* -2 pi f i (/ n)))
					;(exp (complex 0s0 (* -2 pi 2.34 i (/ n))))
				       ))
			 (let ((k_slow :type "complex float*" :init (funcall fun_slow global_a))
			       (k_fast :type "float complex*" :init (funcall fun global_a)))

			   (funcall printf (string ,(format nil "idx     global_a          k_slow           k_fast f=~a\\n" f)))
			   (dotimes (i 16)
			     (let ()
			       (funcall printf (string "%02d   %6.3f+(%6.3f)i %6.3f+(%6.3f)i %6.3f+(%6.3f)i \\n")
					i
					(funcall crealf (aref global_a i))
					(funcall cimagf (aref global_a i))
					(funcall crealf (aref k_slow i))
					(funcall cimagf (aref k_slow i))
					(funcall crealf (aref k_fast i))
					(funcall cimagf (aref k_fast i))))
			     )
			   )))
		       (return 0)))))
    (write-source *main-cpp-filename* "c" code)
    ;(uiop:run-program "clang -Wextra -Wall -march=native -std=c11 -Ofast -ffast-math -march=native -msse2  source/cpu_try.c -g -o source/cpu_try -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -lm")
    (uiop:run-program "gcc -march=native -std=c11 -Ofast -ffast-math -march=native  -ftree-vectorize source/cpu_try.c -o source/cpu_try_gcc -lm")
    (uiop:run-program "gcc -march=native -std=c11 -Ofast -ffast-math -march=native  -ftree-vectorize -S source/cpu_try.c -o source/cpu_try.s")
    ))






