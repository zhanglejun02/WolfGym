# WolfGym

WolfGym evaluates budget-constrained risk intervention in simulated financial
markets. A supervisor observes public market and propagation information and
chooses when and where to intervene. The primary outcome is the reduction in
ordinary investors' counterfactual wealth shortfall, with normal-market
disruption and intervention costs reported alongside it.

## Current experiment

- **Population:** 490 ordinary participants, 10 coordinated manipulators, and
  3 separately counted market makers; 3 assets and 30 trading days.
- **Attack:** a coordinated campaign switches between pump-and-dump,
  finfluencer, spoofing, wash-trading, and WAIT, sharing each participant's
  existing account and resource limits.
- **Actions:** WAIT or a one-day FREEZE of one public asset; each freeze costs
  4, with a total budget of 12.
- **Evaluation:** paired normal/attack conditions, weighted equally, grouped
  by initial society. Train/validation/test contain 1,000/128/256 societies.
- **Input:** public structured observations; model adapters use a five-frame
  history. Reference accounts, fundamentals, attack labels, and evaluation
  outcomes are excluded from model input.

The executable contracts are
[the task and population lock](src/wolfgym/env/combination_v51.json),
[the split manifest](src/wolfgym/benchmark/combination_splits_v1.json), and
[the public input dictionary](src/wolfgym/evaluation/combination_input_dictionary_v2.txt).
Version identifiers in these resources identify frozen protocols; they are not
alternative experiment settings.

## Status

The combination Gym interface, public baselines, PPO and behavior-cloning
development studies, and a small Qwen pilot have been implemented and audited.
Learning methods have not surpassed the selected public rule in the completed
development studies.

As of **2026-10-01**, the fixed-Qwen development evaluation on 32 societies is
paused with **0/64 complete episodes**. The formal multi-model test study has
not run, and the benchmark is not released as a completed leaderboard.
See [experiment status](docs/EXPERIMENT_STATUS.md) for results and outstanding
work.

## Installation and checks

Python 3.10 or newer is required. Experiment runners and the full test suite
use POSIX file locking and support Linux/macOS. From the repository directory:

```bash
python -m pip install -e '.[dev,rl,llm,plot]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m pytest -q
```

The optional groups provide Gym/training, model-client, and plotting dependencies.
Tests use offline model mocks; they do not require an API key or paid requests.
Tests that audit complete local experiment artifacts skip when those artifacts
are unavailable.

## Gym example

This uses a retired engineering society, outside the research splits:

```python
from wolfgym.benchmark.combination import CombinationEpisode
from wolfgym.env.combination import make_combination_env

episode = CombinationEpisode("engineering", 20260925, "attack")
env = make_combination_env(episode)
try:
    observation, info = env.reset()
    observation, reward, terminated, truncated, info = env.step(0)  # WAIT
finally:
    env.close()
```

Complete evaluation runs last 30 days even after a damage event. Evaluation
rewards are zero; terminal counterfactual results are available only to the
trusted evaluation host. Training rewards require explicit opt-in on an allowed
training or engineering episode.

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/wolfgym/core/` | Frozen market, participants, accounts, and random streams |
| `src/wolfgym/env/` | Combination Gym, training interfaces, and verified fast implementations |
| `src/wolfgym/benchmark/` | Society splits, experiment planning, and shared evaluation utilities |
| `src/wolfgym/evaluation/` | Public policies, paired hosts, model adapters, request journals, and audits |
| `scripts/` | Current combination experiment, training, reporting, and recovery commands |
| `tests/` | Core and current experiment regressions; older shared-component settings are test fixtures |
| `data/` | Two small frozen inputs plus ignored local experiment outputs |

- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md): settings, metrics, comparisons, and execution boundaries.
- [Architecture](docs/ARCHITECTURE.md): entry points, dependencies, and extension rules.
- [Market contract](docs/MARKET_CONTRACT.md): financial assumptions and resource accounting.
- [Public observation contract](docs/LANGUAGE_OBSERVATION_CONTRACT.md): exposed fields and information isolation.

Raw trajectories, checkpoints, provider responses, request ledgers, local secrets,
and cleanup snapshots are excluded from Git. Existing experiment manifests and
receipts remain intact locally. Resume requires source-fingerprint review after
repository cleanup; do not overwrite manifests or automatically resend pending
requests. The [data guide](data/README.md) describes the published inputs.

## Scope

This is a controlled simulation of investor protection, with a composite damage
event. It does not certify real financial crises, causal synergy between all
four tactics, or generalization to other populations and market mechanisms.

MIT licensed; see [LICENSE](LICENSE).
