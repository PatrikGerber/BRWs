title Compiling Latex into PDF
:: Takes one command line argument, the name of the tex file you want to compile
set filename=%1
pdflatex --shell-escape %filename%
bibtex %filename%
pdflatex --shell-escape %filename%
pdflatex --shell-escape %filename%
start %filename%.pdf