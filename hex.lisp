
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

