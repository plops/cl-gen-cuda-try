(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-cpp-generator))
(in-package :cl-cpp-generator)

(defmacro e (&body body)
  `(statements (<< "std::cout" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))



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

(progn
  (defparameter *main-cpp-filename*
    (merge-pathnames "stage/cl-gen-cuda-try/source/cpu_try"
		     (user-homedir-pathname)))
  (let* (
	 (code
	  `(with-compilation-unit
	       (include <stdio.h>)
	     (include <complex.h>)

	     (function (fun ((a :type "complex* __restrict__")
			     (n :type int))
			    void)
		       ,(let ((n1 8)
			      (n2 8))
			  `(let (((aref local_a (* ,n1 ,n2)) :type "static complex"))
			     ,@(loop for j below n2 collect
				    `(setf ,@(loop for i below n1 appending
						  `((aref local_a ,(+ (rev i n1)
								      (* j n1)))
						    (aref a ,(+ i (* j n1)))))))
			     (let (((aref line ,n1) :type "static complex"))
			      ,@(loop for j below n2 collect
				     `(statements 
					(dotimes (i ,n1)
					  (setf (aref line i) (aref local_a (+ i ,(* j n1)))))
					(dotimes (k ,n1)
					  (setf (aref local_a (+ k ,(* j n1)))
						(* (aref line k)
						   (funcall cexpf (* k ,(coerce (/ (* 2 j pi)
										   n1)
										'single-float) "1.0fi"
									   )))))))))))	     


	     (decl (((aref global_a 256) :type complex)))
	     (function ("main" ()
			       int)
		       (funcall fun global_a 256)
		       (return 0)))))
    (write-source *main-cpp-filename* "c" code)
    (uiop:run-program "gcc -O3 -march=native source/cpu_try.c -o source/cpu_try")
    (uiop:run-program "gcc -O3 -march=native -S source/cpu_try.c -o source/cpu_try.s")))


(defun flush (a)
  (if (< (abs a) 1e-15)
      0s0
      a))
(defun flush-z (z)
    (let ((a (realpart z))
	  (b (imagpart z)))
      (complex (flush a) (flush b))))


(progn
  (defparameter *main-cpp-filename*
    (merge-pathnames "stage/cl-gen-cuda-try/source/cpu_try"
		     (user-homedir-pathname)))
  (let* (
	 (code
	  `(with-compilation-unit
	       ;; https://news.ycombinator.com/item?id=13147890
	       (raw "//gcc -std=c99 -Ofast -flto -ffast-math -march=skylake -msse2  -ftree-vectorize")
	     ;; https://dendibakh.github.io/blog/2017/10/30/Compiler-optimization-report
	     (raw "//clang -std=c99 -Ofast -flto -ffast-math -march=skylake -msse2 -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize -Rpass-missed=loop-vectorize")
	     (raw "//icc -std=c99 -O2 -D NOFUNCCALL -qopt-report=1 -qopt-report-phase=vec -guide-vec -parallel")
	       (include <stdio.h>)
	     (include <complex.h>)
	     ;(raw "#typedef scomplex float complex")
	     (function (fun ((a :type "float complex* __restrict__")
			     )
			    "float complex*")
		       ,(let ((n1 4)
			      (n2 4))
			  `(let (((aref x (* ,n1 ,n2)) :type "static float complex" :init (list 0.0fi)))
			     (raw "// split 1d into col major n1 x n2 matrix, n1 columns, n2 rows")
			     ;; read columns
			     ,@(loop for j1 below n1 appending
				    (loop for j2 below n2 collect
					 `(setf (aref x (+ ,j1 (* ,n1 ,j2)))
					       (aref a ,(+ j2 (* n2 j1))))))


			     (raw "// dft on each row")
			     (let (((aref s (* ,n1 ,n2)) :type "static float complex" :init (list 0.0fi))
				   ,@(let ((args-seen nil))
				       (loop for j2 below n2 appending
					    (loop for k below n2 when (not (member
									    (mod (+ (/ 1 n2)
										    (/ (* -1 j2 k)
										       n2))
										 1)
									    args-seen))					       collect
						 (progn
						   (push (mod (+ (/ 1 n2) (/ (* -1 j2 k)
										    n2))
							      1)
							 args-seen)
						   `(,(let ((arg (mod (+ (/ 1 n2) (/ (* -1 j2 k)
										    n2))
								     1)))
						       (format nil "wn2_~a_~a"
							       (numerator arg)
							       (denominator arg))) :type "const float complex"
						     :init ,(flush-z (exp (complex 0s0 (* -2 (/ pi n2) j2 k))))))))))
			       ,@(loop for j2 below n2 appending
				      (loop for j1 below n1 collect
					   `(setf (aref s ,(+ j1 (* n1 j2)))
						  (+ ,@(loop for k below n2 collect
							    (if (eq 0 (* j2 k))
								  `(aref x ,(+ j1 (* k n1)))
								  (let ((arg (mod (+ (/ 1 n2) (/ (* -1 j2 k)
												 n2))
										  1)))
								    `(* (aref x ,(+ j1 (* k n1)))
									,(format nil "wn2_~a_~a" (numerator arg)
										 (denominator arg))))))))))

			       (raw "// transpose and elementwise multiplication")
			       (let (((aref z (* ,n1 ,n2)) :type "static float complex" :init (list 0.0fi))
				     ,@(let ((w-seen ()))
					 (loop for j1 below n1 appending
					      (loop for j2 below n2 when (and (/= 0 (* j1 j2))
									      (not 
										  (member (round (* 180000 (/ pi)
												    (phase (exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2))))))))
											  w-seen)))
						 collect
						   (progn
						     (push (round (* 180000 (/ pi)
								     (phase (exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2))))))))
							   w-seen)
						     `(,(format nil "wn_~{~a~}"
								(let ((val (round (* 180000 (/ pi)
										    (phase (exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2))))))))))
								 (list (if (< val 0) "m" "p") (abs val))))
						       :type "const float complex"
						       :init ,(flush-z (exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2))))))))))))
				 ,@(loop for j1 below n1 appending
					(loop for j2 below n2 collect
					     `(setf (aref z ,(+ (* j1 n2) j2))
						    ,(if (eq 0 (* j1 j2))
							 `(aref s ,(+ j1 (* j2 n1)))
							 `(*  (aref s ,(+ j1 (* j2 n1)))
							      ,(format nil "wn_~{~a~}"
							       (let ((val (round (* 180000 (/ pi)
										    (phase (exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2))))))))))
								 (list (if (< val 0) "m" "p") (abs val))))
					;,(exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1 n2)))))
							      )))))
				 (raw "// dft on each row")
				 (let (((aref y (* ,n1 ,n2)) :type "static float complex" :init (list 0.0fi))
				       ,@(let ((seen ()))
					   (loop for j1 below n1 appending
						(loop for j2 below n2 when (and (/= 0 (* j1 j2))
										(not (member (round (* 180000 (/ pi)
												       (phase (exp (complex 0s0 (* -2 pi j1 j2 (/ n1)))))))
											     seen)))
						   collect
						     (progn
						       (push (round (* 180000 (/ pi)
												       (phase (exp (complex 0s0 (* -2 pi j1 j2 (/ n1))))))) seen)
						      `(,(format nil "wn1_~{~a~}"
								 (let ((val (round (* 180000 (/ pi)
										      (phase (exp (complex 0s0 (* -2 pi j1 j2 (/ n1)))))))))
								   (list (if (< val 0) "m" "p") (abs val))))
							 :type "const float complex"
							 :init ,(flush-z (exp (complex 0s0 (* -2 pi j1 j2 (/ (* n1))))))))))))
				   
				   ,@(loop for j1 below n1 appending
					  (loop for j2 below n2 collect
					       `(setf (aref y ,(+ (* j1 n2) j2))
						      (+ ,@(loop for k below n1 collect
								(if (eq 0 (* j1 k))
								    `(aref z ,(+ (* k n2) j2))
								    `(*
								      ,(format nil "wn1_~{~a~}"
								 (let ((val (round (* 180000 (/ pi)
										      (phase (exp (complex 0s0 (* -2 pi j1 k (/ n1)))))))))
								   (list (if (< val 0) "m" "p") (abs val))))
								      ;,(exp (complex 0s0 (* -2 pi j1 k (/ n1))))
								      (aref z ,(+ (* k n2) j2)))))))))
				   (return y)))))))	     


	     (decl (((aref global_a (* 4 4)) :type "float complex")))
	     (function ("main" ()
			       int)
		       (funcall fun global_a)
		       (return 0)))))
    (write-source *main-cpp-filename* "c" code)
    (uiop:run-program "gcc -O3 -march=native source/cpu_try.c -o source/cpu_try")
    (uiop:run-program "gcc -O3 -march=native -S source/cpu_try.c -o source/cpu_try.s")
    ))

