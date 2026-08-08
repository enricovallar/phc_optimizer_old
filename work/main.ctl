(load-module "materials.ctl")
(load-module "parity_functions.ctl")
(load-module "shapes.ctl")
(load-module "wyckoff.ctl")
(load-module "custom_nonbloch_output.ctl")
(load-module "lattices.ctl")

(set! deterministic? true)
(set! num-bands 8)
(set! resolution 32)
(define-param h 0.5)
(define-param r1 0.2)
(define-param r2 0.1)

; Dynamic mode execution flags (can be overridden via command line k=v pairs)
(define-param run-te? true)
(define-param run-tm? true)
(define-param run-zeven? false)
(define-param run-zodd? false)

(define slab-mat (make dielectric (epsilon 9.46)))
(define air-mat (make dielectric (epsilon 1)))

(set! geometry-lattice (make-hexagonal-lattice no-size))
(define background-slab
  (make block (size (vector3 1e20 1e20 h))
              (center (vector3 0 0 0))
              (material slab-mat)))
(define shapes-1a
  (map (lambda (pos) (make-param-cylinder pos h r1 air-mat))
       (get-C6v-1a)))
(define shapes-2b
  (map (lambda (pos) (make-param-cylinder pos h r2 air-mat))
       (get-C6v-2b)))
(set! geometry (make-superposition background-slab (list shapes-1a shapes-2b)))

(define-param kmag 0.1)
(set! k-points (interpolate 10 (get-hex-path-circular kmag)))
(display-kpath-labels hex-labels-circular)

; Conditionally execute solvers based on dynamic flags
(if run-te? (run-te))
(if run-tm? (run-tm))
(if run-zeven? (run-zeven))
(if run-zodd? (run-zodd))
