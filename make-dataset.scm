#!/usr/bin/env -S guix build -f
!#

;;; This guix script is used to preprocess the idf file through each version
;;; converter, then through the ExpandObjects program and finally through the
;;; ConvertInputFormat to turn it into epJSON.

(use-modules
 (srfi srfi-1)
 (guix gexp)
 (guix build utils)
 (terramorpha packages energyplus))

(define versions
  '(;; "9.0.0"
    ;; "9.1.0"
    ;; "9.2.0"
    ;; "9.3.0"
    ;; "9.4.0"
    "9.5.0"
    "9.6.0"
    "22.1.0"
    "22.2.0"
    "23.1.0"
    "23.2.0"
    "24.1.0"
    ))

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

(define (update-all-the-way filename)
  (define name (basename filename))
  (define transitions (map (lambda (from to)
                             (lambda (file)
                               (computed-file (string-append (car (string-split name #\.)) "-" to ".idf")
                                              (update-version file
                                                              #:from from
                                                              #:to to)
                                              #:local-build? #f)))

                           (list-head versions (1- (length versions)))
                           (cdr versions)))

(fold (lambda (updater file) (updater file))
      (local-file filename) transitions))

(define prof
  (getenv "GUIX_LOAD_PROFILE"))

(define all-idf-files
  (find-files (string-append prof "/share/energyplus/buildings/") ".*\\.idf"))

(define all-the-updated-files
  (map update-all-the-way all-idf-files))

(define new-dataset
  (computed-file
   "energyplus-dataset-family-1a-24.1.0"
   (with-imported-modules
    '((guix build utils))
    #~(begin
        (use-modules (guix build utils))
        (for-each
         (lambda (filename)
            (install-file filename (string-append #$output "/share/energyplus/buildings/"))))
        '#$all-the-updated-files))))

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

(define (convert-format file)
  (computed-file
   "converted.json"
   (with-imported-modules
    '((guix build utils))
    #~(begin
        (use-modules (guix build utils))
        (copy-file #$file "./in.idf")
        (invoke #$(file-append energyplus "/bin/ConvertInputFormat") "./in.idf")
        (copy-file "./in.epJSON" #$output)))
   #:local-build? #f))


;; (convert-format
;;  (expand-objects
;;   (update-all-the-way "./file.idf")))
new-dataset
