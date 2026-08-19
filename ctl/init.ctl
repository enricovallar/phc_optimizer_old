; ==============================================================================
; init.ctl
; Base loader helper and default solver configuration for MPB modular library
; ==============================================================================

; Module loader helper
(define (load-module filename)
  (if (file-exists? filename)
      (primitive-load filename)
      (primitive-load-path filename)))

; Helper to check if current simulation is a 3D slab (finite supercell height sz)
(define (is-slab?)
  (and (defined? 'sz) (not (equal? (primitive-eval 'sz) no-size))))

; Default MPB Solver Parameters (safe type-checks for scalar vs vector3)
(if (and (number? resolution) (= resolution 10)) (set! resolution 32))
(if (and (number? num-bands) (= num-bands 1)) (set! num-bands 8))
(set! deterministic? true)

; Default resolution in z-direction for 3D slabs
(define-param res-z 16)
(define-param res_z 16)

; Dynamic mode execution flags (can be overridden via CLI k=v pairs or case scripts)
(define-param run-te? false)
(define-param run-tm? false)
(define-param run-zeven? false)
(define-param run-zodd? false)

; Dynamic symmetry display flags
(define-param display-symmetry? false)
(define-param display_symmetry? false)

; Dynamic group velocity display flags
(define-param display-group-velocity? false)
(define-param display_group_velocity? false)

; Dynamic k-path override (can be set via CLI k=v pair to evaluate ONLY Gamma point)
(define-param only-gamma? false)
(define-param only_gamma? false)

; Dynamic delta-k group velocity calculation flags
(define-param delta-k-mode? false)
(define-param delta_k_mode? false)
(define-param delta-k 0.01)
(define-param delta_k 0.01)

