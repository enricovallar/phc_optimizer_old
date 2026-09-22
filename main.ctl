; ==============================================================================
; MPB Control File: Modular Photonic Crystal Simulation
; ==============================================================================

(load-module "materials.ctl")
(load-module "parity_functions.ctl")
(load-module "shapes.ctl")
(load-module "wyckoff.ctl")
(load-module "custom_nonbloch_output.ctl")
(load-module "lattices.ctl")

; Geometric parameters (overridden by command line / BO runner)
(define-param h 0.5)
(define-param r1 0.25)
(define-param r2 0.15)
(define-param resolution 64)
(define-param num-bands 12)

; Material definitions
(define matrix-mat (make dielectric (epsilon 12.0)))
(define atom-mat (make dielectric (epsilon 1.0)))

; Lattice geometry (C4v square lattice or C6v triangular lattice)
(set! geometry-lattice (make-square-lattice no-size))

; Background slab and parametric Wyckoff cylinder shapes
(define background-slab
  (make block (size (vector3 1e20 1e20 h))
              (center (vector3 0 0 0))
              (material matrix-mat)))

(define shapes-1
  (map (lambda (pos) (make-param-cylinder pos h r1 atom-mat))
       (get-C4v-1a)))

(define shapes-2
  (map (lambda (pos) (make-param-cylinder pos h r2 atom-mat))
       (get-C4v-1b)))

(set! geometry (make-superposition background-slab (list shapes-1 shapes-2)))

; High-symmetry k-path trajectory
(define-param kmag 0.1)
(set! k-points (interpolate 10 (get-sq-path-circular kmag)))
(display-kpath-labels sq-labels-circular)

; Override k-points to Gamma point (0 0 0) if only-gamma? flag is set during BO
(if (or only-gamma? only_gamma?)
    (set! k-points (list (vector3 0 0 0))))

(define-param delta-k 0.01)
; Override k-points for small delta-k group velocity run
(if (or delta-k-mode? delta_k_mode?)
    (set! k-points (list (vector3 delta-k 0 0))))

; Solver callbacks
(if run-te? (run-solver-with-callbacks run-te))
(if run-tm? (run-solver-with-callbacks run-tm))
(if run-zeven? (run-solver-with-callbacks run-zeven))
(if run-zodd? (run-solver-with-callbacks run-zodd))
