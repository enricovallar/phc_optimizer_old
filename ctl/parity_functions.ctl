; --- C4v Definitions for Square Lattice: basis1=(1,0), basis2=(0,1) ---
(define C4-s (matrix3x3 (vector3 0 1 0) (vector3 -1 0 0) (vector3 0 0 1)))
(define C2-s (matrix3x3 (vector3 -1 0 0) (vector3 0 -1 0) (vector3 0 0 1)))
(define sv-s (matrix3x3 (vector3 1 0 0) (vector3 0 -1 0) (vector3 0 0 1)))  
(define sd-s (matrix3x3 (vector3 0 1 0) (vector3 1 0 0) (vector3 0 0 1)))  

(define (display-symmetries-c4v)
  (if (vector3= current-k (vector3 0 0 0))
      (begin
        (print "SYM_DATA_START_" parity "
")
        (map (lambda (b)
               (print parity "," b 
                      ",C4=" (compute-symmetry b C4-s (vector3 0 0 0)) 
                      ",C2=" (compute-symmetry b C2-s (vector3 0 0 0)) 
                      ",sv=" (compute-symmetry b sv-s (vector3 0 0 0)) 
                      ",sd=" (compute-symmetry b sd-s (vector3 0 0 0)) "
"))
             (arith-sequence 1 1 num-bands))
        (print "SYM_DATA_END_" parity "
"))))


; --- C6v Definitions for Hexagonal Lattice: basis1=(1,0), basis2=(0.5, 0.866) ---
(define C6-h (matrix3x3 (vector3 0 1 0) (vector3 -1 1 0) (vector3 0 0 1))) 
(define C3-h (matrix3x3 (vector3 -1 1 0) (vector3 -1 0 0) (vector3 0 0 1)))
(define C2-h (matrix3x3 (vector3 -1 0 0) (vector3 0 -1 0) (vector3 0 0 1))) 
(define sv-h (matrix3x3 (vector3 1 0 0) (vector3 1 -1 0) (vector3 0 0 1)))  
(define sd-h (matrix3x3 (vector3 0 1 0) (vector3 1 0 0) (vector3 0 0 1)))   

(define (display-symmetries-c6v)
  (if (vector3= current-k (vector3 0 0 0))
      (begin
        (print "SYM_DATA_START_" parity "
") ; Unique start tag
        (map (lambda (b)
               (print parity "," b 
                      ",C6=" (compute-symmetry b C6-h (vector3 0 0 0)) 
                      ",C3=" (compute-symmetry b C3-h (vector3 0 0 0)) 
                      ",C2=" (compute-symmetry b C2-h (vector3 0 0 0)) 
                      ",sv=" (compute-symmetry b sv-h (vector3 0 0 0)) 
                      ",sd=" (compute-symmetry b sd-h (vector3 0 0 0)) "
"))
             (arith-sequence 1 1 num-bands))
        (print "SYM_DATA_END_" parity "
")))) ; Unique end tag

