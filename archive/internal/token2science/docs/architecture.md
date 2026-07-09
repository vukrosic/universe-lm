# Architecture

`token2science` turns donated AI tokens into reproducible science.

## Core Idea

The system is organized around one simple object model:

```text
Goal -> Task -> Run
```

- `Goal` is a research question with a measurable pass bar.
- `Task` is one claimable unit of work inside a goal.
- `Run` is one execution attempt and its evidence artifact.

The model is intentionally small so the whole workflow can live in GitHub and
stay auditable by default.

## GitHub As The Backend

GitHub is treated as the entire backend, not just the code host.

- Issues are the task board.
- Pull requests are submissions.
- GitHub Actions is the verifier.
- GitHub accounts are identity.

This means planning, claiming, submitting, and verifying all happen in one
public system with a shared audit trail.

## Verification Chain

Every submission moves through a layered check:

1. Config-hash receipt
   - The submitted run records what config was used.
   - The verifier checks that the config file in the task folder hashes to the
     recorded `config_hash`.
2. CI reproduce
   - GitHub Actions reruns the experiment from the task folder.
   - The accepted value must match the submitted value within tolerance.
3. K-worker confirmation
   - Independent workers reproduce the same result.
   - Once at least `K` distinct workers support the same run, the result is
     confirmed.

The chain is strict on provenance before it is generous about credit.

## Trust And Reputation

Trust is earned, not assumed.

- A submission with a matching hash but no reproduction is not enough.
- A reproduced run is better, but still only one machine.
- Confirmation requires independent agreement from multiple workers.

Reputation can be derived from confirmed work, successful replication history,
and consistency across submissions. A worker who repeatedly produces valid,
reproducible runs should become more trusted than one who submits noisy or
non-reproducible results.

The long-term goal is to make reputation useful for routing better tasks,
granting higher-value work, and pricing bounties fairly.

## Phase Roadmap

### Phase 1 - Bring Your Own Compute

- Single repo.
- Volunteers run their own compute.
- GitHub issues, PRs, and Actions handle the full loop.
- Verification is cheap and immediate enough for the demo tier.

### Phase 2 - K-Replication, Reputation, Pooling, Bounties

- Add K-replication as the norm for confirmation.
- Attach reputation to workers and confirmed results.
- Support pooled token donation.
- Use reputation and confirmation status to unlock bounties.

### Phase 3 - Hosted Fleet And Sponsor Dashboards

- Provide a hosted worker fleet.
- Route tasks automatically to available compute.
- Add sponsor dashboards for visibility into donated tokens, verified work,
  and funded outcomes.

## Design Constraint

Keep the system reproducible first. If a result cannot be re-run from the
recorded task state, it does not count as science.
