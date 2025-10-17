# LLM-Driven Validation Pipeline

This repository now contains a three-stage workflow that lets you stress-test BranchMind against a sliding-window baseline using large language models for both scenario generation and automated scoring.

## 1. Generate Scenario Scripts

Use `benchmarks/validation/generate_scenarios.py` (or `python -m benchmarks.validation.generate_scenarios`) to produce multi-threaded conversation scripts in the expected JSON format.

```bash
python3 benchmarks/validation/generate_scenarios.py \
  --output evaluation/scenarios.json \
  --count 6
```

Key flags:

- `--prompt-path`: customise the generator instructions (default `prompts/generator_prompt.md`).
- `--split-dir`: optionally emit one JSON file per scenario.
- `--model`: override the Azure OpenAI generator model (`GENERATOR_MODEL` / `TASK_MODEL` fallback).
- `--api-key`, `--azure-endpoint`, `--api-version`: override environment defaults as needed.

## 2. Run BranchMind vs Baseline

Execute both systems across each scenario with `benchmarks/validation/run_evaluation.py` (or `python -m benchmarks.validation.run_evaluation`).

```bash
python3 benchmarks/validation/run_evaluation.py \
  --scenarios evaluation/scenarios.json \
  --output-dir evaluation/runs
```

What happens:

1. Each scenario is copied to `evaluation/runs/<scenario_id>/scenario.json`.
2. BranchMind is instantiated with a fresh state file per scenario and run turn by turn.
3. The baseline (“sliding window + periodic summary”) assistant is run in lockstep.
4. Both transcripts are saved as `result_branchmind.json` and `result_baseline.json`.
5. `results_index.json` summarises all outputs for downstream tooling.

Useful options:

- `--resume`: skip scenarios that already have transcripts.
- `--baseline-window-turns`, `--baseline-summary-trigger`: tweak baseline memory behaviour.
- `--branchmind-*` flags: override BranchMind’s CM/task/summariser models if needed.
- `--baseline-model`, `--baseline-summarizer-model`: override baseline models.
- `--branchmind-state-root`: store BranchMind state files in a custom directory.

## 3. Blind Judgement

Ask another LLM to provide an impartial ranking with `benchmarks/validation/run_judgement.py` (or `python -m benchmarks.validation.run_judgement`).

```bash
python3 benchmarks/validation/run_judgement.py \
  --results-dir evaluation/runs \
  --output-dir evaluation/judgement
```

Features:

- Randomises which transcript is “assistant A/B” per scenario to keep evaluations blind.
- Uses `prompts/evaluator_prompt.md` to enforce a consistent scoring rubric.
- Persists per-scenario judgements plus an aggregated `summary.json` (averages, win counts, preference rate).

### Environment Variables

All scripts respect the same Azure OpenAI environment variables unless overridden via CLI flags:

- `API_KEY`, `AZURE_ENDPOINT`, `API_VERSION`
- `TASK_MODEL`, `CM_MODEL`, `SUMMARIZER_MODEL`
- `GENERATOR_MODEL`, `BASELINE_MODEL`, `BASELINE_SUMMARIZER`, `EVALUATOR_MODEL`

## Suggested Next Steps

1. Review `evaluation/judgement/summary.json` for high-level performance deltas.
2. Inspect `preference_reason` and `justification` fields inside each scenario’s judgement for qualitative insights.
3. Calibrate the scenario generator prompt to emphasise domains you care about most.
4. Extend `SlidingWindowBaseline` or plug in alternative baselines to broaden comparisons.
