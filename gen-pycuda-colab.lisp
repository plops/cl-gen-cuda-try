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
	    (function (dot ((result :type int*)
			  (a :type int*)
			  (b :type int*))
                                   "__global void")
	   (let ((i :type "const int" :init threadIdx.x)
		 )
	     (setf result (+ result (aref a i) (aref b i)))))
	  (raw " ")))))
  #.(in-package :cl-py-generator)
  (defparameter *path* "/home/martin/stage/cl-gen-cuda-try/")
  (defparameter *code-file* "pycuda_colab")
  (defparameter *source* (format nil "~a/source/~a" *path* *code-file*))
  (defparameter *code-facts*
    `((10 "")
      ))

  (let* ((code `(do0
		 "! pip install pycuda"			
		 (imports ((drv pycuda.driver)
			   pycuda.tools
			   pycuda.autoinit
			   pycuda.compiler
			   (np numpy)))
		 (setf mod (pycuda.compiler.SourceModule
			    (string3 ,cl-cpp-generator::*cl-program*))
		       multiply_them (mod.get_function (string "dot")))
		 (setf a (np.random.randint 1 20 5)
		       b (np.random.randint 1 20 5)
		       result 0)
		 ("dot" (drv.Out result)
		      (drv.In a)
		      (drv.In b)
		      :block (tuple 5 1 1))
		 (print result)
		 )))
    (write-source *source* code)
    (sb-ext:run-program "/usr/bin/xclip" (list (format nil "~a.py" *source*)))))

