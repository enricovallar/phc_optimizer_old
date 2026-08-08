; ==============================================================================
; example.ctl
; Master execution script for MPB
; ==============================================================================

; 1. Load Custom Modules
(primitive-load-path "materials.ctl")
(primitive-load-path "parity_functions.ctl")
(primitive-load-path "shapes.ctl")
(primitive-load-path "wyckoff.ctl")
(primitive-load-path "custom_nonbloch_output.ctl")
(primitive-load-path "lattices.ctl")

; 2. Global MPB Parameters
(set! deterministic? true)
(set! num-bands 10)
(set! resolution 32)
(define-param h 0.5)

; Dynamic mode execution flags (can be overridden via CLI k=v pairs)
(define-param run-te? true)
(define-param run-tm? true)
(define-param run-zeven? false)
(define-param run-zodd? false)

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

; Conditionally execute solvers
(if run-te? (run-te))
(if run-tm? (run-tm))
(if run-zeven? (run-zeven))
(if run-zodd? (run-zodd))
