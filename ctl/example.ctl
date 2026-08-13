; ==============================================================================
; example.ctl
; Master execution script for MPB
; ==============================================================================

; 1. Load Custom Modules
(load-module "materials.ctl")
(load-module "parity_functions.ctl")
(load-module "shapes.ctl")
(load-module "wyckoff.ctl")
(load-module "custom_nonbloch_output.ctl")
(load-module "lattices.ctl")

; 2. Case-Specific Geometry Parameters
(define-param h 0.5)

; 3. Define Materials
(define slab-mat (make dielectric (epsilon 9.46)))
(define air-mat (make dielectric (epsilon 1)))

; 4. Define Lattice
(set! geometry-lattice (make-hexagonal-lattice no-size))

; 5. Construct Geometry
(define background-slab
  (make block (size (vector3 1 1 h))
              (center (vector3 0 0 0))
              (material slab-mat)))

(define shapes-1a
  (map (lambda (pos) (make-daisy pos h 100 6 0.05 0.02 air-mat))
       (get-C6v-1a)))

(define shapes-6d
  (map (lambda (pos) (make-param-cylinder pos h 0.15 air-mat))
       (get-C6v-6d 0.2)))

(set! geometry
  (make-superposition background-slab (list shapes-1a shapes-6d)))

; 6. Define K-Points & Labels for Plotting
(define-param kmag 0.1)
(set! k-points (interpolate 10 (get-hex-path-circular kmag)))
(display-kpath-labels hex-labels-circular)

; Override k-points to ONLY Gamma point if only-gamma? flag is set
(if (or only-gamma? only_gamma?)
    (set! k-points (list (vector3 0 0 0))))


; 7. Conditionally execute solvers based on dynamic flags (initialized in init.ctl)
(if run-te? (run-solver-with-callbacks run-te))
(if run-tm? (run-solver-with-callbacks run-tm))
(if run-zeven? (run-solver-with-callbacks run-zeven))
(if run-zodd? (run-solver-with-callbacks run-zodd))
