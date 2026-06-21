# arXiv submission bundle

LaTeX source for the paper *Quantized Recurrent State for Mamba Inference*.

## Build
1. Generate figures: `python make_figures.py` (writes PDFs to figures/).
2. Compile: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## arXiv upload
Upload main.tex, references.bib, the generated main.bbl, and figures/*.pdf. Do NOT upload the PDF; arXiv compiles from source. Set author/affiliation in main.tex and choose a license at submission time.
