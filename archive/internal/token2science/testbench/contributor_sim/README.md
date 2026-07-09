# Contributor simulation

Simulate real people doing AI research through token2science - claiming tasks,
running experiments (against the mock backend, so no GPU and no waiting),
submitting results, and publishing papers with their name on them. Use it to
test the whole human-facing workflow end to end at full speed.

## Pieces

- `capped_launch.sh` - the safety layer. A launcher that enforces a GLOBAL cap
  (`T2S_SIM_MAX`, default 4) on concurrent simulation sessions, so that agents
  spawning agents can never explode. Wraps the fire-and-forget codex launcher,
  which auto-kills each session when its agent prints `FINAL:` or times out.
- `persona_contributor.md` - the prompt for one simulated newcomer who does the
  full loop: claim -> run (mock) -> submit -> paper -> release.
- `persona_coordinator.md` - the prompt for an agent that spawns N contributors
  through `capped_launch.sh` and tallies the outcome.

## Safety model

- All spawning goes through `capped_launch.sh`. The cap is counted from live
  tmux session names (`simu-*`), so it holds globally even under recursion.
- Sessions self-terminate on `FINAL:` (both personas end with that line) or at
  the timeout, so nothing lingers.
- It is a soft cap: a rare race could start one extra session, never a runaway.
  Lower `T2S_SIM_MAX` to be stricter.

## How to launch (later)

One contributor (smoke test, 1 session):

```
T2S_SIM_MAX=1 bash token2science/testbench/contributor_sim/capped_launch.sh \
  contrib-1 "$(cat token2science/testbench/contributor_sim/persona_contributor.md)
  Your handle: sim-user-1." 600
```

A full wave via the coordinator (respects the cap):

```
T2S_SIM_MAX=4 /Users/vukrosic/.claude/skills/launch-codex-tmux/scripts/launch_and_wait.sh \
  simu-coordinator "$(cat token2science/testbench/contributor_sim/persona_coordinator.md)
  Simulate N=6 contributors." 1800
```

## What a good run proves

- contributors complete the loop without human help (the workflow is usable),
- papers get generated with the right author handles (the incentive works),
- claim collisions = 0 (exclusive leasing holds under real concurrent load),
- none of it touched a GPU (the mock backend stood in for compute).
