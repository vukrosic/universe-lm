---
name: latex-paper-generator
description: Generates a high-quality LaTeX research paper from markdown content, including plots and architecture diagrams, and compiles it into a PDF.
---

# LaTeX Paper Generator Skill

You are a **Scientific Typesetter**. Your goal is to transform a markdown research paper into a publication-ready LaTeX document (`paper.tex`) and compile it into a PDF (`paper.pdf`).

## Workflow

1.  **Read Source Content**:
    - Identify the final markdown paper in `docs/papers/`.
    - Identify any result plots (e.g., `plots/*.png` or similar).

2.  **Generate `paper.tex`**:
    - Use the standard `article` class or a conference template (e.g., NeurIPS style if requested).
    - **Header**: Title, Authors (Vuk Rosić and Gemini), Abstract.
    - **Body**: Convert markdown sections to LaTeX equivalents (`\section`, `\subsection`, etc.).
    - **Math**: Ensure all equations are properly formatted in LaTeX (`\begin{equation}`, `$ ... $`).
    - **Tables**: Convert markdown tables to proper LaTeX `tabular` environments.
    - **Figures**: Include any result plots using `\includegraphics`.
    - **Citations**: If references exist, format them properly with `\bibitem` or BibTeX.

3.  **Compile PDF**:
    - Ensure LaTeX is installed: `apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-science` (if needed).
    - Run `pdflatex -interaction=nonstopmode paper.tex` from the `docs/papers/` directory.
    - Run it twice to resolve references if necessary.
    - Check for compilation errors and fix them.

4.  **Output**:
    - `docs/papers/paper.pdf`: The final compiled document.
    - `docs/papers/paper.tex`: The source code.

## How to use this skill
1.  **Input**: The markdown paper file path (e.g., `docs/papers/<topic>.md`).
2.  **Action**: Generate `paper.tex` and compile.
3.  **Result**: A PDF file at `docs/papers/paper.pdf`.
