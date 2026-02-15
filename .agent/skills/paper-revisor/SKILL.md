---
name: paper-revisor
description: Iterates on a research paper draft based on reviewer feedback. Addresses weaknesses and clarifies complex sections.
---

# Paper Revisor Skill

You improve a paper draft based on specific reviewer feedback. Every weakness must be addressed or acknowledged.

## Revision Protocol

### 1. Map Feedback to Sections
For each reviewer weakness/comment, identify the exact section(s) that need changes.

### 2. Priority Order
1.  **Statistical issues** (missing error bars, no ablations, overclaimed significance) — fix first
2.  **Missing experiments** (trivial baselines, wall-clock comparisons) — add or acknowledge as limitations
3.  **Mathematical issues** (wrong terminology, missing definitions) — fix
4.  **Writing issues** (jargon, tone, clarity) — fix last

### 3. Revision Rules
- If the reviewer says the result is within noise: **Do not argue.** Add more seeds, or downgrade the claim.
- If the reviewer asks for an ablation that hasn't been done: Either run it (preferred) or add it to "Limitations & Future Work" with an honest acknowledgment.
- If the reviewer flags invented terminology: Replace it with standard terms. Don't defend bad naming.
- If the reviewer says wall-clock is worse: Report it prominently, don't hide it in a footnote.

### 4. Consistency Check
After revisions:
- Do the abstract claims still match the experimental results?
- Are all numbers in the abstract also in the results table?
- Do the conclusions follow from the statistical evidence?

## Output

The revised paper with a changelog:

```markdown
## Revision Changelog
| Reviewer Point | Section Changed | What Was Done |
|---------------|-----------------|---------------|
```
