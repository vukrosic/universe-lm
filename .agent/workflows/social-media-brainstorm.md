# Diverse Research Question Brainstorming
---
description: A workflow to generate a simple, diverse list of technical research questions based on the current codebase.
---

Follow these steps to generate a diverse list of research questions in a single autonomous pass:

## Stage 1: Idea Generation
1.  Use the `ai-research-innovator` skill to analyze the current codebase and knowledge.
2.  Generate 5-7 original, diverse research ideas spanning architecture, optimization, and training dynamics.

## Stage 2: Technical Sanity Check
1.  Use `idea-reviewer` to evaluate the technical soundness and diversity of the generated ideas.
2.  Filter out trivial or redundant questions.
3.  Select the **top 5 most diverse** ideas.

## Stage 3: Question Refinement
1.  Use the `social-media-writer` (Research Question Generator) skill to convert the ideas into a simple, one-sentence list of questions.
2.  Ensure no paragraphs, hooks, or platform-specific content are included.

## Stage 4: Final Report & Persistence
1.  Save the finalized list of questions to a `.txt` file in `docs/research/questions/` (e.g., `research_questions_YYYYMMDD.txt`).
2.  Present the finalized list to the user immediately as a simple plain text list.
3.  **DO NOT wait for user input or approval**. Complete the generation and display the results in a single response.

**Rule**: This is a non-interactive, one-pass workflow. Proceed autonomously until the final list is complete.
