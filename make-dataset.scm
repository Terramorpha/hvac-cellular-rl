;;; This guix script is used to preprocess the idf file through each version
;;; converter, then through the ExpandObjects program and finally through the
;;; ConvertInputFormat to turn it into epJSON.

(use-modules
 (srfi srfi-1)
 (guix gexp)
 (guix build utils)
 (guix store)
 (guix derivations))

(for-each
 (lambda (p)
   (add-to-load-path p))
 (string-split (getenv "GUIX_PACKAGE_PATH")
               #\:))

(use-modules (terramorpha packages energyplus))

(define energyplus
  energyplus)

(define* (update-version file #:key from to)
  (define (turn-version v)
    (string-map (lambda (c) (if (eq? c #\.)
                             #\-
                             c))
                v))
  (define fromidd (string-append "V" (turn-version from) "-Energy+.idd"))
  (define toidd (string-append "V" (turn-version to) "-Energy+.idd"))
  (with-imported-modules
   '()
   #~(begin
       (copy-file (string-append #$energyplus "/PreProcess/IDFVersionUpdater/" #$fromidd)
                  #$fromidd)
       (copy-file (string-append #$energyplus "/PreProcess/IDFVersionUpdater/" #$toidd)
                  #$toidd)
       (copy-file #$file "./file.idf")
       ;; ça va toujours exit avec 1 (à cause d'un bug fortran}, mais le fichier
       ;; est quand même bon.
       (system* (string-append
                 #$energyplus
                 "/PreProcess/IDFVersionUpdater/Transition-V"
                 #$(turn-version from)
                 "-to-V"
                 #$(turn-version to))
               "./file.idf")
       (copy-file "./file.idfnew"
                  #$output))))

(define (update-all-the-way file versions)
  (define transitions (map (lambda (from to)
                             (lambda (file)
                               (computed-file (string-append "version-" to ".idf")
                                              (update-version file
                                                              #:from from
                                                              #:to to)
                                              #:local-build? #f)))

                           (list-head versions (1- (length versions)))
                           (cdr versions)))

 (fold (lambda (updater file) (updater file))
       file
       transitions))

(define (expand-objects file)
  (computed-file
   "expanded.idf"
   (with-imported-modules
    '((guix build utils))
    #~(begin
        (use-modules (guix build utils))
        (copy-file #$file "./in.idf")
        (invoke #$(file-append energyplus "/bin/ExpandObjects"))
        (copy-file "./expanded.idf" #$output)))))

(define (to-json file)
  (computed-file
   "converted-to.json"
   (with-imported-modules
    '((guix build utils))
    #~(begin
        (use-modules (guix build utils))
        (copy-file #$file "./in.idf")
        (invoke #$(file-append energyplus "/bin/ConvertInputFormat") "./in.idf")
        (copy-file "./in.epJSON" #$output)))
   #:local-build? #f))

(define (from-json file)
  (computed-file
   "converted-to.idf"
   (with-imported-modules
    '((guix build utils))
    #~(begin
        (use-modules (guix build utils))
        (copy-file #$file "./in.epJSON")
        (invoke #$(file-append energyplus "/bin/ConvertInputFormat") "./in.epJSON")
        (copy-file "./in.idf" #$output)))
   #:local-build? #f))

(define versions
  '(;; "9.0.0"
    ;; "9.1.0"
    ;; "9.2.0"
    ;; "9.3.0"
    ;; "9.4.0"
    ;; "9.5.0"
    ;; "9.6.0"
    ;; "22.1.0"
    ;; "22.2.0"
    "23.1.0"
    "23.2.0"
    "24.1.0"))

(define alburquerque
  (to-json
   (update-all-the-way
    (file-append energyplus-dataset-commercial-small-office
                 "/share/energyplus/buildings/ASHRAE901_OfficeSmall_STD2004_Albuquerque.idf")
    '("22.1.0" "22.2.0" "23.1.0" "23.2.0" "24.1.0"))))

(define crawlspace
  (to-json
   (expand-objects
    (update-all-the-way
     (file-append energyplus-dataset-residential-1a
                  "/share/energyplus/buildings/US+MF+CZ1AWH+elecres+crawlspace+IECC_2006.idf")
     '("9.5.0"
       "9.6.0"
       "22.1.0"
       "22.2.0"
       "23.1.0"
       "23.2.0"
       "24.1.0")))))


;; (to-json
;;  (update-all-the-way (from-json (local-file filename))
;;                      versions))


(define (dataset)
  (file-union
   "dataset"
   `(("alburquerque.epJSON" ,alburquerque)
     ("crawlspace.epJSON" ,crawlspace))))

(define store (open-connection))

(define-values (drv conn) ((lower-object (dataset)) store))

(build-derivations conn (list drv))

(define p (derivation-output-path (cdar (derivation-outputs drv))))

(use-modules (guix build utils))

(copy-recursively p "./buildings"
                  #:follow-symlinks? #t
                  #:keep-permissions? #f)
