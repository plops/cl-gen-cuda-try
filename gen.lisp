(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-cpp-generator))
(in-package :cl-cpp-generator)
(defmacro e (&body body)
  `(statements (<< "std::cout" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))



;; ssh -p 1235 localhost -L 5900:localhost:5900 -L 2221:10.1.10.3:22

(progn
  (defparameter *main-cpp-filename*
    (merge-pathnames "stage/cl-gen-cuda-try/source/cuda_try"
		     (user-homedir-pathname)))
  (let* ((code `(with-compilation-unit
		    (include <stdio.h>)
		  
		 (raw "// https://www.youtube.com/watch?v=Ed_h2km0liI CUDACast #2 - Your First CUDA C Program")
		 (raw "// https://github.com/NVIDIA-developer-blog/cudacasts/blob/master/ep2-first-cuda-c-program/kernel.cu")
		 (function ("vector_add" ((a :type "int*")
					  (b :type "int*")
					  (c :type "int*")
					  (n :type "int"))
					 "__global__ void")
			   
			   (let ((i :type int :init threadIdx.x)
				 )
			     (if (< i n)
				 (statements
				  (setf (aref c i) (+ (aref a i)
						      (aref b i)))))))
		 (enum (N 1024))
		 (function ("main" ()
				   "int")
			   
			   (let (,@(loop for e in '(a b c) collect
					`(,e :type int* :init (funcall malloc (* N (sizeof int)))))
				 ,@(loop for e in '(a b c) collect
					`(,(format nil "d_~a" e) :type int*))
				 				 ))))))
    (write-source *main-cpp-filename* "cpp" code)))
