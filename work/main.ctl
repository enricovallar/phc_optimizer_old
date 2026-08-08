(load-module "materials.ctl")
(load-module "parity_functions.ctl")
(load-module "shapes.ctl")
(load-module "wyckoff.ctl")
(load-module "custom_nonbloch_output.ctl")
(load-module "lattices.ctl")

(set! deterministic? true)
(set! num-bands 8)
(set! resolution 16)
(define-param h 0.5)

(define slab-mat (make dielectric (epsilon 9.46)))
(define air-mat (make dielectric (epsilon 1)))

(set! geometry-lattice (make-hexagonal-lattice no-size))
(define background-slab
  (make block (size (vector3 1 1 h))
              (center (vector3 0 0 0))
              (material slab-mat)))
(define shapes-1a
  (map (lambda (pos) (make-daisy pos h 100 6 0.05 0.02 air-mat))
       (get-C6v-1a)))
(set! geometry (make-superposition background-slab (list shapes-1a)))

(define-param kmag 0.1)
(set! k-points (interpolate 10 (get-hex-path-circular kmag)))
(display-kpath-labels hex-labels-circular)

(run-te)
