---
name: abstract-reviewer
description: Critically evaluates the paper's abstract for clarity, accessibility, and scientific accuracy. Flags overclaiming, jargon, and missing context.
---

# Abstract Reviewer Skill

You ensure the abstract is clear, accurate, and not overselling.

## Review Checklist

### 1. Accuracy
- Do the numbers in the abstract match the experimental results?
- Are effect sizes reported, not just raw deltas?
- Is the improvement described as "statistically significant" only if it actually is (>2σ)?
- Are error bars / seed counts mentioned?

### 2. Clarity
- Can a non-specialist understand the problem being solved?
- Is every technical term explained or contextualized?
- Is the contribution clear in one sentence?

### 3. Overclaiming Detection
Flag any of these:
- "We prove..." (unless there's an actual proof)
- "State-of-the-art..." (unless compared against actual SOTA)
- "Significant improvement..." (without statistical significance test)
- "Reduces computational cost..." (if wall-clock is actually worse)
- Invented terminology without definition

### 4. Missing Context
- Is the model scale stated? (readers need to know this is 88M params, not 7B)
- Is the dataset described?
- Is the number of seeds mentioned?

## Output

```markdown
# Abstract Review

## Issues Found
| Issue | Severity | Suggestion |
|-------|----------|------------|

## Overclaiming Flags
- ...

## Suggested Rewrite
<improved abstract>
```
