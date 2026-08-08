; ==============================================================================
; init.ctl
; Base loader helper for MPB modular configuration library
; ==============================================================================

(define (load-module filename)
  (if (file-exists? filename)
      (primitive-load filename)
      (primitive-load-path filename)))
