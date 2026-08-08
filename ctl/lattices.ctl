; ---------------------------------------------------------
; Square Lattice
; ---------------------------------------------------------
(define (make-square-lattice supercell-height)
  (make lattice 
    (size 1 1 supercell-height)
    (basis1 (vector3 1.0 0.0 0.0))
    (basis2 (vector3 0.0 1.0 0.0))))

; Square k-points
(define sq-G (vector3 0.0 0.0 0.0))
(define sq-X (vector3 0.5 0.0 0.0))
(define sq-M (vector3 0.5 0.5 0.0))

; ---------------------------------------------------------
; Hexagonal Lattice
; ---------------------------------------------------------
(define (make-hexagonal-lattice supercell-height)
  (make lattice 
    (size 1 1 supercell-height)
    (basis1 (vector3 1.0 0.0 0.0))
    (basis2 (vector3 0.5 (/ (sqrt 3) 2) 0.0))))

; Hexagonal k-points
(define hex-G (vector3 0.0 0.0 0.0))
(define hex-M (vector3 0.0 0.5 0.0))
(define hex-K (vector3 -1/3 1/3 0.0))

; ---------------------------------------------------------
; Oblique Lattice
; ---------------------------------------------------------
; Pass a1 and a2 as (vector3 ...) objects
(define (make-oblique-lattice a1 a2 supercell-height)
  (make lattice 
    (size 1 1 supercell-height)
    (basis1 a1)
    (basis2 a2)))

; Oblique k-points
(define obl-G (vector3 0.0 0.0 0.0))
(define obl-K (vector3 0.5 0.0 0.0))
(define obl-M (vector3 0.5 0.5 0.0))


; ---------------------------------------------------------
; K-Point High-Symmetry Labels
; ---------------------------------------------------------
(define label-G "Gamma")
(define label-X "X")
(define label-M "M")
(define label-K "K")


; ---------------------------------------------------------
; K-Paths & Corresponding High-Symmetry Labels for Plotting
; ---------------------------------------------------------

; --- Gamma is centered ---

; Square: X -> Gamma -> M -> X
(define sq-path-centered 
  (list sq-X sq-G sq-M sq-X))
(define sq-labels-centered 
  (list label-X label-G label-M label-X))

; Hexagonal: K -> Gamma -> M -> K
(define hex-path-centered 
  (list hex-K hex-G hex-M hex-K))
(define hex-labels-centered 
  (list label-K label-G label-M label-K))

; Oblique: K -> Gamma -> M -> K
(define obl-path-centered 
  (list obl-K obl-G obl-M obl-K))
(define obl-labels-centered 
  (list label-K label-G label-M label-K))


; --- Starting with Gamma ---

; Square: Gamma -> X -> M -> Gamma
(define sq-path-starting 
  (list sq-G sq-X sq-M sq-G))
(define sq-labels-starting 
  (list label-G label-X label-M label-G))

; Hexagonal: Gamma -> K -> M -> Gamma
(define hex-path-starting 
  (list hex-G hex-K hex-M hex-G))
(define hex-labels-starting 
  (list label-G label-K label-M label-G))

; Oblique: Gamma -> K -> M -> Gamma
(define obl-path-starting 
  (list obl-G obl-K obl-M obl-G))
(define obl-labels-starting 
  (list label-G label-K label-M label-G))


; --- True Circular Space, Gamma-centered ---

; Helper function: maps a reciprocal direction to Cartesian, 
; normalizes its length to 1, scales it by the given radius, 
; and maps it back to reciprocal space for MPB.
(define (circular-k-point k-dir radius)
  (cartesian->reciprocal 
    (vector3* radius 
              (unit-vector3 (reciprocal->cartesian k-dir)))))

; Square: (radius towards X) -> Gamma -> (radius towards M)
(define (get-sq-path-circular radius)
  (list (circular-k-point sq-X radius) 
        sq-G 
        (circular-k-point sq-M radius)))
(define sq-labels-circular 
  (list "-X" label-G "M"))

; Hexagonal: (radius towards K) -> Gamma -> (radius towards M)
(define (get-hex-path-circular radius)
  (list (circular-k-point hex-K radius) 
        hex-G 
        (circular-k-point hex-M radius)))
(define hex-labels-circular 
  (list "K" label-G "M"))

; Oblique: (radius towards K) -> Gamma -> (radius towards M)
(define (get-obl-path-circular radius)
  (list (circular-k-point obl-K radius) 
        obl-G 
        (circular-k-point obl-M radius)))
(define obl-labels-circular 
  (list "K" label-G "M"))


; ---------------------------------------------------------
; Helper Function to Output Labels for Python Plotting
; ---------------------------------------------------------
(define (display-kpath-labels labels)
  (print "KPATH_LABELS: " labels "\n"))