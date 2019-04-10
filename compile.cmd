title Compiling Latex into PDF
:: Takes one command line argument, the name of the tex file you want to compile
set filename=%1
pdflatex %filename%
rem bibtex %filename%
rem pdflatex %filename%
rem pdflatex %filename%
start %filename%.pdf