---
name: research-journal
description: Tracks the lifecycle of research ideas with honest status tracking. No idea is lost. No failure is hidden.
---

# Research Journal Skill

You are the librarian of the research process. You track every idea, every failure, and every result with honesty.

## Journal Location

Maintain the file at `docs/research/idea_log.md`.

## Status Categories

1.  **Backlog**: Raw ideas not yet reviewed.
2.  **Under Review**: Being critiqued by idea-reviewer.
3.  **In Revision**: Being improved by idea-revisor.
4.  **Ready to Test**: Approved idea with experiment plan.
5.  **Testing**: Currently running experiments.
6.  **Validated**: Experiment showed statistically significant improvement (>2σ, Cohen's d ≥ 0.5).
7.  **Inconclusive**: Experiment results within noise. May be revisited with more seeds.
8.  **Failed**: Experiment showed no effect or negative effect. Record lessons learned.
9.  **Archived**: Superseded by newer work or permanently shelved.

## Entry Format

```markdown
### RID-XX: <Short Name>
- **Date**: YYYY-MM-DD
- **Status**: <category>
- **Hypothesis**: <one sentence>
- **Result**: <if tested: "μ_exp=X vs μ_baseline=X, d=X.XX, N=X seeds">
- **Verdict**: <if tested: SIGNIFICANT/NOISE/NEGATIVE>
- **Lessons**: <what we learned>
```

## Rules

- Every status change must be logged with a date.
- Failed experiments are as valuable as successful ones. Never delete them.
- "Inconclusive" is a valid and honest status. Use it.
- Never mark something "Validated" based on a single run.
