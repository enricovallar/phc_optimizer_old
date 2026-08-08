; ==============================================================================
; wyckoff.ctl
; ==============================================================================

; ----------------------------------------------------------------------
; Standard C6v (Hexagonal / p6mm) Wyckoff Positions
; Returns lists of (vector3 ...) coordinates
; ----------------------------------------------------------------------

(define (get-C6v-1a)
  (list (vector3 0 0 0)))

(define (get-C6v-2b)
  (list (vector3 1/3 1/3 0)
        (vector3 2/3 2/3 0)))

(define (get-C6v-3c)
  (list (vector3 1/2 0 0)
        (vector3 0 1/2 0)
        (vector3 1/2 1/2 0)))

(define (get-C6v-6d x-dist)
  (let ((x x-dist)
        (neg-x (- x-dist)))
    (list (vector3 x 0 0)
          (vector3 0 x 0)
          (vector3 neg-x x 0)
          (vector3 neg-x 0 0)
          (vector3 0 neg-x 0)
          (vector3 x neg-x 0))))

(define (get-C6v-6e x-dist)
  (let ((x x-dist)
        (neg-x (- x-dist))
        (two-x (* 2 x-dist))
        (neg-two-x (* -2 x-dist)))
    (list (vector3 x x 0)
          (vector3 neg-x two-x 0)
          (vector3 neg-two-x x 0)
          (vector3 neg-x neg-x 0)
          (vector3 x neg-two-x 0)
          (vector3 two-x neg-x 0))))


; ----------------------------------------------------------------------
; Standard C4v (Square / p4mm) Wyckoff Positions
; Returns lists of (vector3 ...) coordinates
; ----------------------------------------------------------------------

(define (get-C4v-1a)
  (list (vector3 0 0 0)))

(define (get-C4v-1b)
  (list (vector3 1/2 1/2 0)))

(define (get-C4v-2c)
  (list (vector3 1/2 0 0)
        (vector3 0 1/2 0)))

(define (get-C4v-4d x-dist)
  (let ((x x-dist)
        (neg-x (- x-dist)))
    (list (vector3 x x 0)
          (vector3 neg-x x 0)
          (vector3 neg-x neg-x 0)
          (vector3 x neg-x 0))))

(define (get-C4v-4e x-dist)
  (let ((x x-dist)
        (neg-x (- x-dist)))
    (list (vector3 x 0 0)
          (vector3 0 x 0)
          (vector3 neg-x 0 0)
          (vector3 0 neg-x 0))))


; ----------------------------------------------------------------------
; Linear Superposition Method
; ----------------------------------------------------------------------
; Combines a bulk geometry block (like a slab) with multiple lists 
; of generated shapes into a single flat list for MPB's geometry variable.

(define (make-superposition bulk-geom shape-lists)
  ; (apply append ...) flattens a list of lists into a single list
  (append (list bulk-geom) (apply append shape-lists)))