(load-module "materials.ctl")
(load-module "parity_functions.ctl")
(load-module "shapes.ctl")
(load-module "wyckoff.ctl")
(load-module "custom_nonbloch_output.ctl")
(load-module "lattices.ctl")

; Case-specific geometric parameters
(define-param h 0.5)
(define-param r1 0.2)
(define-param r2 0.1)

(define matrix-mat (make dielectric (epsilon 9.47)))
(define atom-mat (make dielectric (epsilon 1)))

(set! geometry-lattice (make-square-lattice no-size))
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

(define-param kmag 0.1)
(set! k-points (interpolate 10 (get-sq-path-circular kmag)))
(display-kpath-labels sq-labels-circular)

; Override k-points to ONLY Gamma point if only-gamma? flag is set
(if (or only-gamma? only_gamma?)
    (set! k-points (list (vector3 0 0 0))))

; Conditionally execute solvers based on dynamic flags (initialized in init.ctl)
(if run-te? (run-solver-with-callbacks run-te))
(if run-tm? (run-solver-with-callbacks run-tm))
(if run-zeven? (run-solver-with-callbacks run-zeven))
(if run-zodd? (run-solver-with-callbacks run-zodd))
