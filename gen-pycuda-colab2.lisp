(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator"))



#+nil(setf *features* (union *features* '())) 
#+nil(setf *features* (set-difference *features* '()))




(progn
  (progn
    #.(in-package #:cl-cpp-generator)
    (progn
 (defun pairup (l)
   (cond ((null l) nil)
         ((null (cdr l)) (list l))   
         (t (let ((p (pairup (cddr l))))
	      (if p
		  (list (list (car l) (cadr l))
			p)
		  (list (car l) (cadr l)))))))
 #+nil
 (pairup '(1 2 3 4))

 (defun c+ (rest ;&rest rest
		    )
   (labels ((frob (l)
	      (cond ((null l) 0)
		    ((not (listp l)) l)
		    ((and (listp l) (null (cdr l))) (frob (car l)))
		    ((listp l) `(funcall cuCaddf ,(frob (car l)) ,(frob (cdr l))))
		    (t (break "error ~a" l)))))
     (frob (pairup rest))))
 #+nil
 (c+ 1 2 3 4))
    (defun flush (a)
  (if (< (abs a) 1e-15)
      0s0
      a))
  (defun flush-z (z)
    (let ((a (realpart z))
	  (b (imagpart z)))
      `(funcall CMPLXF ;complex
		,(flush a) ,(flush b)
		)))
  
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
      (1/4 `(funcall cuCmulf (funcall ;;__builtin_complex ;;
					;make_cuFloatComplex ;
		CMPLXF
		(* -1 (funcall cimagf ,e))
		(funcall crealf ,e))))
      (-1/4 `(funcall cuCmulf (funcall ;; __builtin_complex ;;
		 ;make_cuFloatComplex ;
		 CMPLXF
		 (funcall cimagf ,e)
		 (* -1 (funcall crealf ,e)))))
      (3/4 `(funcall cuCmulf (funcall ;; __builtin_complex ;;
					; make_cuFloatComplex ;
		CMPLXF
		 (funcall cimagf ,e)
		 (* -1 (funcall crealf ,e)))))
      (t `(funcall cuCmulf ,e
		   ,(format nil "w~a" (twiddle-arg-name j k n))))))
  #+nil(defun c+ (&rest rest)
    `())
    (defparameter *cl-program*
      (cl-cpp-generator::beautify-source
       `(with-compilation-unit
	    (raw " ")
	  ;(include "/usr/local/cuda-10.0/targets/x86_64-linux/include/cuComplex.h")
	  (raw "#include <cuComplex.h>")
	  (raw "#define cimagf(x) cuCimagf(x)")
	  (raw "#define crealf(x) cuCrealf(x)")
	  (raw "#define CMPLXF(x,y) make_cuFloatComplex((x),(y))")
	  (raw "typedef cuFloatComplex complex;")
	    ,(let* ((n1 3)
		    (n2 7)
		    ;(N2 64)
		     (n (* n1 n2))
		    ;(dft (format nil "dft_~a" n))
		     (fft (format nil "fft_~a_~a_~a" n n1 n2)))
		(flet ((row-major (a x y)
			 `(aref ,a ,(+ (* 1 x) (* n1 y))))
		       (col-major (a x y)
			 `(aref ,a ,(+ (* n2 x) (* 1 y)))))
		  `(statements
		    (raw "// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm")
		    #+nil (function (,dft ((a :type "complex* __restrict__"))
				   "complex*")
			     (setf a (funcall __builtin_assume_aligned a 64)) ;; tell compiler that argument ins 64byte aligned
			     (let (((aref y ,n) :type "static alignas(64) complex" :init (list 0.0fi)))
			       #+memset (funcall memset y 0 (* ,n (funcall sizeof "complex")))
			       (dotimes (j ,n)
				 (dotimes (k ,n)
				   (setf (aref y j) (+ (aref y j)
						       (* (aref a k)
							  (funcall cexpf (* "1.0fi" ,(* -2 pi (/ n)) j k)))))))
			       (return y)))
		   
		   (function (,fft
			      (;(x :type "complex* __restrict__")
			       (dst :type "complex*")
			       (src :type "complex*"))
			      "__global__ void")
			     (raw "// n1 DFTs of size n2 in the column direction")
			     (let ((i :type "const int" :init threadIdx.x)
				   (x :type "complex*" :init (+ src (* ,(* n1 n2) i)))
				   ((aref x1 ,(* n1 n2)) :type "complex")
				   ,@(let ((args-seen (list 0 1/4 -1/4 3/4 1/2)))
				       (loop for k2 below n2 appending ;; column
					    (loop for n2_ below n2
					       when (not (member (twiddle-arg n2_ k2 n2) args-seen))
					       collect
						 (progn
						   (push (twiddle-arg n2_ k2 n2) args-seen)
						   `(,(format nil "w~a" (twiddle-arg-name n2_ k2 n2)) :type "const complex"
						      :init ,(flush-z (exp (complex 0s0 (* -2 (/ pi n2) n2_ k2))))))))))
			       ,@(loop for k2 below n2 appending 
				      (loop for n1_ below n1 collect
					   `(setf ,(row-major 'x1 n1_ k2)
						  ,(c+ 
						   (loop for n2_ below n2 collect
							  (twiddle-mul (row-major 'x n1_ n2_)
								       n2_ k2 n2))))))
			       (raw "// multiply with twiddle factors and transpose")
			       (let (((aref x2 ,(* n1 n2)) :type "complex")
				     ,@(let ((args-seen (list 0 1/4 -1/4 3/4 1/2)))
					 (loop for k2 below n2 appending
					      (loop for n1_ below n1
						 when (not (member (twiddle-arg n1_ k2 n) args-seen))
						 collect
						   (progn
						     (push (twiddle-arg n1_ k2 n) args-seen)
						     `(,(format nil "w~a" (twiddle-arg-name n1_ k2 n)) :type "const complex"
							:init ,(flush-z (exp (complex 0s0 (* -2 (/ pi n) n1_ k2))))))))))
			       
				 ,@(loop for k2 below n2 appending 
					(loop for n1_ below n1 collect
					     `(setf ,(col-major 'x2 n1_ k2)
						    ,(twiddle-mul (row-major 'x1 n1_ k2)
								  n1_ k2 n))))
				 (raw "// another dft")
				 (let ((;(aref x3 ,(* n1 n2)) :type "float complex"
					x3 :type "complex*" :init (+ dst (* ,(* n1 n2) i)))
				       ,@(let ((args-seen (list 0 1/4 -1/4 3/4 1/2)))
					   (loop for k1 below n1 appending ;; column
						(loop for n1_ below n1
						   when (not (member (twiddle-arg n1_ k1 n1) args-seen))
						   collect
						     (progn
						       (push (twiddle-arg n1_ k1 n1) args-seen)
						       `(,(format nil "w~a" (twiddle-arg-name n1_ k1 n1)) :type "const complex"
							  :init ,(flush-z (exp (complex 0s0 (* -2 (/ pi n1) n1_ k1))))))))))
				 
				   ,@(loop for k2 below n2 appending 
					  (loop for k1 below n1 collect
					       `(setf ,(col-major 'x3 k1 k2)
						      (+ ,@(loop for n1_ below n1 collect
								(twiddle-mul (col-major 'x2 n1_ k2)
									     n1_ k1 n1)))))))))))))
	    (raw " ")))))
  #.(in-package :cl-py-generator)
  (defparameter *path* "/home/martin/stage/cl-gen-cuda-try/")
  (defparameter *code-file* "pycuda_colab2")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *code-facts*
    `((10 "")
      ))

  (let* ((code `(do0
		 ;"! pip install pycuda"			
		 (imports ((drv pycuda.driver)
			   pycuda.tools
			   pycuda.autoinit
			   pycuda.compiler
			   (np numpy)))
		 (setf mod (pycuda.compiler.SourceModule
			    (string3 ,cl-cpp-generator::*cl-program*))
		       fft_21_3_7 (mod.get_function (string "fft_21_3_7")))
		 (setf N2 64
		       n1 3
		       n2 7
		       src (dot (np.random.randn (list (* n1 n2) n2))
			      (astype np.complex64))
		       result (np.zeros_like a))
		 (fft_21_3_7 (drv.Out dst)
			     (drv.In src)
			     :block (tuple N2 1 1))
		 (print dst)
		 )))
    (write-source *source* code)
    (sb-ext:run-program "/usr/bin/xclip" (list (format nil "~a.py" *source*)))))

