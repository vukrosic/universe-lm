---
name: latex-paper-generator
description: Generates a high-quality LaTeX research paper from markdown content, including plots and architecture diagrams, and compiles it into a PDF.
---

# LaTeX Paper Generator Skill

You are a **Scientific Typesetter**. Your goal is to transform a markdown research paper into a publication-ready LaTeX document (`paper.tex`) and compile it into a PDF (`paper.pdf`).

## Workflow

1.  **Read Source Content**:
    - Identify the final markdown paper (e.g., `docs/papers/curvature_aware_muon.md`).
    - Identify the key result plot (e.g., `plots/winner_100m.png` or similar).

2.  **Generate `paper.tex`**:
    - Use the standard `article` class or a conference template (e.g., NeurIPS style if requested, otherwise standard academic).
    - **Header**: Title, Authors (Vuk Rosić and Gemini), Abstract.
    - **Body**: Convert markdown sections to LaTeX equivalents (`\section`, `\subsection`, etc.).
    - **Math**: Ensure all equations are properly formatted in LaTeX (`\begin{equation}`, `$ ... $`).
    - **Figures**:
        - Include the key result plot using `\includegraphics`.
        - **architecture drawing**: Create a TikZ diagram illustrating the "Adaptive Gating" mechanism (e.g., Gradient -> Frobenius Norm -> Gating Logic -> Newton-Schulz Steps).
    - **Citations**: If references exist, format them properly.

3.  **Compile PDF**:
    - Run `pdflatex -interaction=nonstopmode paper.tex`.
    - Run it twice to resolve references if necessary.
    - Check for compilation errors.

4.  **Output**:
    - `paper.pdf`: The final compiled document.
    - `paper.tex`: The source code.

## TikZ Diagram Instructions
When creating the architecture diagram, focus on the **OGO Logic**:
- Nodes: Input Gradient ($G$), Frobenius Norm ($||G||_F$), Threshold Check ($\tau$), Fast Path ($N=4$), Safe Path ($N=5$), Output ($G_{orth}$).
- Arrows showing the flow of data.
- Styling: Professional, clean lines, academic aesthetic.

## How to use this skill
1.  **Input**: The markdown paper file path.
2.  **Action**: Generate `paper.tex` and compile.
3.  **Result**: A PDF file ready for download/viewing.
