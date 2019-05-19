(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload "cl-py-generator")
  (ql:quickload "cl-cpp-generator"))



#+nil(setf *features* (union *features* '())) 
#+nil(setf *features* (set-difference *features* '()))


(progn
  (progn
    #.(in-package #:cl-cpp-generator)
    
    (defparameter *cl-program*
      (cl-cpp-generator::beautify-source
       `(with-compilation-unit
	    (raw " ")
	    (function (cu_mul ((result :type float*)
			  (a :type float*)
			  (b :type float*))
                                   "__global__ void")
	   (let ((i :type "const int" :init threadIdx.x)
		 )
	     (setf (aref result i) (* (aref a i) (aref b i)))))
	  (raw " ")))))
  #.(in-package :cl-py-generator)
  (defparameter *path* "/home/martin/stage/cl-gen-cuda-try/")
  (defparameter *code-file* "pycuda_colab")
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
		       cu_mul (mod.get_function (string "cu_mul")))
		 (setf n 400
		       a (dot (np.random.randn n)
			      (astype np.float32))
		       b (dot (np.random.randn n)
			      (astype np.float32))
		       result (np.zeros_like a))
		 (cu_mul (drv.Out result)
		      (drv.In a)
		      (drv.In b)
		      :block (tuple n 1 1))
		 (print result)
		 )))
    (write-source *source* code)
    (sb-ext:run-program "/usr/bin/xclip" (list (format nil "~a.py" *source*)))))

