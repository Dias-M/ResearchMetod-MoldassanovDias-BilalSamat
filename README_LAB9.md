# Lab 9: IEEE paper skeleton (Overleaf)

Upload: Overleaf -> New Project -> Upload Project -> this zip. Compiler: pdfLaTeX, main file `main.tex`.

Structure: `main.tex` (IEEEtran, packages), `sections/01_introduction.tex`, `sections/02_methodology.tex`, `sections/03_results.tex` and `04_discussion.tex` (placeholders for Task 4), `references.bib` (Zotero export, cleaned: no abstract/file fields).

Switch to ACM: replace the class line with `\documentclass[sigconf]{acmart}` and replace the IEEEtran author/abstract/keywords blocks with the acmart commands (`\author`, `\affiliation`, `\begin{abstract}`, `\keywords`); bibliography style `ACM-Reference-Format`.

Audit (checklist of the lab): no space before `\cite` (all `~\cite`), no straight quotes, no hard-coded numbers (all via `\ref`/`\eqref`), every label is defined and every reference resolves, every cited key exists in the .bib, no BibTeX warnings in the test build.

To check before submission: venue of `lain_phishing_2021`, `ho_understanding_2025`, `thapa_phishing_2025` (the "verify" notes were removed from the .bib), the sentence about the gap in the Introduction (it is limited to the studies we reviewed), and the author affiliation lines.
