; ---------------------------------------------------------
; shapes.ctl
; ---------------------------------------------------------

; -------------------------
; Helper: Daisy Vertices
; -------------------------
(define (get-daisy-vertices num-pts m-val r0-val rd-val)
  (map (lambda (i)
         (let* ((phi (* 2.0 3.141592653589793 (/ i num-pts)))
                (r (+ r0-val (* rd-val (cos (* m-val phi)))))
                (x (* r (cos phi)))
                (y (* r (sin phi))))
           (cartesian->lattice (vector3 x y 0))))
       (iota num-pts)))

; -------------------------
; Constructor: Daisy Shape
; -------------------------
; Returns a single prism object centered at the specified Wyckoff position.
(define (make-daisy center-vec h num-pts m-val r0-val rd-val mat)
  (make prism 
    (vertices (get-daisy-vertices num-pts m-val r0-val rd-val))
    (height h)
    (center center-vec)
    (material mat)))

; -------------------------
; Constructor: Single Cylinder
; -------------------------
; Returns a single cylinder object centered at the specified Wyckoff position.
(define (make-param-cylinder center-vec h r mat)
  (make cylinder 
    (radius r)
    (height h)
    (center center-vec)
    (material mat)))