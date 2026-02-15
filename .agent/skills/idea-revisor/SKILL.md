---
name: idea-revisor
description: Iterates on research ideas based on reviewer critique. Addresses weaknesses honestly, simplifies overcomplicated proposals, and ensures testability.
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
- Use standard names for standard concepts (e.g., "gradient norm threshold" not "Spectral Energy Gating")

### 3. Add What's Missing
- **Trivial baselines**: If the reviewer asked "what about just always using N=4?", add that as a mandatory ablation.
- **Failure predictions**: State what results would disprove the idea.
- **Overhead analysis**: Add wall-clock cost estimates.

### 4. Strengthen the Math
- If the math was called weak, add formal definitions, not more analogies.
- If terminology was called misleading, fix it. Don't defend bad names.

### 5. Constrain the Scope
- All experiments must be feasible at ≤1B tokens.
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

## Experiment Plan
<With ablations and trivial baselines>

## Failure Criteria
<What results would kill this idea>
```
