# BranchMind Workspace

This repository houses the BranchMind dialogue manager together with two benchmarking
pipelines:

1. **Validation** – short/medium multi-thread scenarios for regression-style testing.
2. **Stress** – long-form scenarios (50–100 turns) that probe depth, breadth, span, and
   control-complexity limits.

The tree-structured manager itself lives in the `branchmind/` package; both pipelines call
into that package while adding their own orchestration logic.

---

## Repository Layout

```
branchmind/                # Core dialogue manager implementation
benchmarks/
  common.py                # Shared data models + Azure client helpers
  validation/              # Standard evaluation pipeline
    generate_scenarios.py
    run_evaluation.py
    run_full_history_baseline.py
    run_judgement.py
    __init__.py
  stress/                  # Stress-testing pipeline
    run_stress_suite.py
    run_stress_analysis.py
    build_scenario.py      # Iterative LLM-backed scenario generator (outline → chunks)
    prompts/
    scenarios/             # Outlines + soon JSON fixtures
docs/                      # Additional workflow docs (e.g. llm_validation_pipeline.md)
history/                   # Previous main_* implementations kept for reference
```

---

## Environment & Configuration

All Python entry points assume **Python 3.12** (see `.python-version`). A virtualenv or
`uv` environment is recommended. Required packages include `python-dotenv` and the
official `openai` SDK; install via your preferred tool, e.g.:

```bash
uv pip install -r requirements.txt         # if you add one
# or
pip install python-dotenv openai httpx
```

For stress-report visualisations install `matplotlib`:

```bash
pip install matplotlib
```

### Azure OpenAI Credentials

Set the usual BranchMind variables via environment or `.env`:

- `API_KEY`
- `AZURE_ENDPOINT`
- `API_VERSION`
- `TASK_MODEL`, `CM_MODEL`, `SUMMARIZER_MODEL`
- Optional overrides: `GENERATOR_MODEL`, `BASELINE_MODEL`, `BASELINE_SUMMARIZER`,
  `EVALUATOR_MODEL`, `STRESS_EVALUATOR_MODEL`

**Never commit live credentials.** The repo currently contains a `.env` example; replace
values before pushing or move to secrets management.

### PYTHONPATH Notes

Scripts can be executed in two ways:

1. Module form (preferred): `python -m benchmarks.validation.run_evaluation ...`
2. Direct file execution: `python benchmarks/validation/run_evaluation.py ...`

Each script prepends the project root to `sys.path` so both styles work when executed from
the repository root.

---

## Core Dialogue Manager

The entry point remains `branchmind/main.py`. You can run an interactive session via:

```bash
python -m branchmind.main --api-key ... (other flags)
```

Key CLI flags:

- `--api-key`, `--azure-endpoint`, `--api-version`
- `--cm-model`, `--task-model`, `--summarizer-model`
- `--state-path` (JSON persistence, empty string disables)
- `--retry-attempts`, `--retry-delay`

Internally the manager now exposes instrumentation hooks (`DialogueObserver`,
`LLMCallRecord`, `TreeSnapshot`) that the stress suite reuses.

---

## Validation Pipeline

The validation flow mirrors `docs/llm_validation_pipeline.md` (paths updated):

1. **Generate Scenarios**

   ```bash
   python -m benchmarks.validation.generate_scenarios \
     --output evaluation/scenarios.json \
     --count 6
   ```

   Key flags: `--prompt-path`, `--split-dir`, `--model`.

2. **Run BranchMind vs Baselines**

   ```bash
   python -m benchmarks.validation.run_evaluation \
     --scenarios evaluation/scenarios.json \
     --output-dir evaluation/runs
   ```

   Useful flags:

   - `--baseline-window-turns`, `--baseline-summary-trigger`
   - `--branchmind-*` / `--baseline-*` model overrides
   - `--resume`, `--scenario-limit`

3. **Blind Judgement**

   ```bash
   python -m benchmarks.validation.run_judgement \
     --results-dir evaluation/runs \
     --output-dir evaluation/judgement
   ```

   Evaluator prompt lives in `prompts/evaluator_prompt.md`.

---

## Stress Testing Pipeline

Assets and scripts reside in `benchmarks/stress/`.

### 1. Scenario Preparation

- Hand-author or LLM-generate long scenarios (50–100 turns). Outlines for “AI 创业日志”
  and “全能管家” live in `benchmarks/stress/scenarios/`.
- Use `benchmarks/stress/build_scenario.py` to iteratively expand an outline via LLM:

  ```bash
  python -m benchmarks.stress.build_scenario \
    benchmarks/stress/scenarios/concierge_extreme_outline.md \
    stress_scenarios/concierge.json \
    --target-turns 90 \
    --chunk-size 12 \
    --verbose
  ```

  The tool first drafts a scenario plan (`*.plan.json`), then requests dialogue chunks in
  batches while enforcing thread IDs and schema validation. Use `--dry-run` to inspect
  the plan, `--resume` to continue after partial generation, and `--skip-metadata` to
  postpone rubric creation.

### 2. Execute Assistants

```bash
python -m benchmarks.stress.run_stress_suite \
  --scenarios benchmarks/stress/scenarios/your_scenario.json \
  --output-dir stress_results \
  --assistants branchmind sliding_window full_history
```

Options:

- `--baseline-window-turns`, `--baseline-summary-trigger`
- `--full-history-system-prompt`
- `--sleep-between-assistants`, `--sleep-between-turns`
- `--retry-attempts`, `--retry-delay`

Outputs per scenario:

- `branchmind/`, `sliding_window/`, `full_history/` subdirectories with
  `transcript.json`, `metrics.jsonl`, `summary.json`, and BranchMind state/telemetry.

### 3. Automated Analysis

```bash
python -m benchmarks.stress.run_stress_analysis \
  --results-dir stress_results \
  --output-dir stress_results/analysis
```

Uses `benchmarks/stress/prompts/stress_evaluator_prompt.md` to score task success,
forgetfulness, and (for BranchMind) controller accuracy. Produces per-run JSON plus an
aggregate summary.

### 4. Reporting & Visualisation

```bash
python -m benchmarks.stress.reporting \
  --results-dir stress_results \
  --output-dir stress_results/report
```

Generates per-scenario summaries (`summary.json`) and, when `matplotlib` is available,
renders latency, token usage, and BranchMind tree-depth charts under
`stress_results/report/<scenario_id>/`. An aggregate `aggregate_summary.json` captures
assistant averages across all processed scenarios.

> **Note:** Both stress scripts call live Azure OpenAI endpoints. In restricted network
> sandboxes they will fail with `httpx.ConnectError`. Ensure outbound access or mock the
> client.

---

## Additional Notes

- Historical implementations remain in `history/` for reference; they are not imported by
  default.
- Token usage and latency telemetry is captured for all assistants; BranchMind additionally
  records tree snapshots per turn.
- If you add reporting notebooks, prefer storing them under `benchmarks/stress/` or
  `benchmarks/validation/` to keep concerns separated.
- Before committing, scrub any generated data containing user content or secrets. Consider
  adding `.gitignore` entries for `evaluation/`, `stress_results/`, and similar output dirs.

Happy testing!
