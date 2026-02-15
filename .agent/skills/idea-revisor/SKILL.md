---
name: idea-revisor
description: Iterates on research ideas based on reviewer critique. Addresses weaknesses honestly, simplifies overcomplicated proposals, and ensures testability at 1M tokens.
---

# Idea Revisor Skill

You take a critiqued idea and make it stronger by addressing the actual weaknesses — not by adding more jargon.

## Revision Protocol

### 1. Triage the Feedback
Classify each critique as:
- **Fatal flaw**: The idea is fundamentally broken. Pivot or drop.
- **Addressable weakness**: Can be fixed with better math, clearer experiments, or simpler formulation.
- **Style issue**: Terminology, naming, or presentation.

### 2. Simplify, Don't Complicate
If the reviewer says the idea is overcomplicated:
- Strip it down to the minimal mechanism
- Remove invented terminology
- Use standard names for standard concepts

### 3. Add What's Missing
- **Trivial baselines**: If the reviewer asked for one, add it as a mandatory ablation.
- **Failure predictions**: State what results would disprove the idea.
- **Overhead analysis**: Add wall-clock cost estimates.
- **Implementation flag**: Specify the exact CLI flag name (e.g., `--use_spectral_gate`).

### 4. Strengthen the Math
- If the math was called weak, add formal definitions, not more analogies.
- If terminology was called misleading, fix it.

### 5. Constrain the Scope
- All experiments must be feasible at **1M tokens**.
- Remove any claims about "scaling to 1T tokens" or "production readiness."

## Output

```markdown
# Research Proposal V2: <Name>

## Changes from V1
| V1 Issue | V2 Fix |

## Revised Hypothesis
<Clear, falsifiable statement>

## Revised Method
<Simplified description with correct terminology>

## Implementation
- Flag: `--<flag_name>`
- Files to modify: <list>
- Lines of code: <estimated>

## Experiment Plan
- Scale: 1M tokens
- Seeds: 42, 137, 256 (minimum)
- Baseline: 5-seed variance from `docs/research/baseline_variance_1M.md`
- Success: Cohen's d ≥ 0.5

## Failure Criteria
<What results would kill this idea>
```
