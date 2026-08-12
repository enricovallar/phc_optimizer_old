; ==============================================================================
; init.ctl
; Base loader helper and default solver configuration for MPB modular library
; ==============================================================================

; Module loader helper
(define (load-module filename)
  (if (file-exists? filename)
      (primitive-load filename)
      (primitive-load-path filename)))

; Default MPB Solver Parameters
; Note: MPB's C runtime initializes num-bands=1 and resolution=10 by default.
; If not overridden via command-line k=v arguments, set standard defaults:
(if (= num-bands 1) (set! num-bands 8))
(if (= resolution 10) (set! resolution 32))
(set! deterministic? true)

; Dynamic mode execution flags (can be overridden via CLI k=v pairs or case scripts)
(define-param run-te? true)
(define-param run-tm? true)
(define-param run-zeven? false)
(define-param run-zodd? false)
