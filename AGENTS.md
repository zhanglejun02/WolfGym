# Project working rules

- Prioritize readable, minimal code: descriptive names, explicit control flow,
  small cohesive functions, and comments explaining non-obvious assumptions.
- Reuse existing implementations. Do not add speculative abstractions, duplicate
  pipelines, or compatibility layers without a concrete requirement.
- Preserve existing user changes. Verify behavioral changes with focused tests;
  report what was and was not tested.
- Keep the research scope in financial-market risk intervention. Follow
  `docs/EXPERIMENT_PROTOCOL.md`; do not expand task taxonomy, domains,
  language generation, or training infrastructure just to improve benchmark scores.
- A research plan is not authorization to launch experiments. Keep experiments
  bounded and distinguish protocol fixes from population/model tuning.
- Treat `src/wolfgym/env/combination_v51.json` and
  `src/wolfgym/benchmark/combination_splits_v1.json` as the executable task and
  split contracts. Preserve their bound source bytes; do not regenerate hashes
  to hide behavioral changes or bypass experiment-manifest checks.
- Maintain the stable documents in `docs/`; update current settings and status
  in place instead of adding dated or numbered protocol documents.
- Keep full experiment outputs, request journals, pending requests, and costs
  intact in ignored local data. Never resume a paused or paid batch as part of
  repository maintenance.
