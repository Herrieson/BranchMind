# Stress Testing Framework

This directory houses assets and utilities for running long-form stress scenarios against BranchMind and baseline dialogue managers.

## Layout

- `scenarios/` – canonical JSON fixtures (50–100 turns) plus human-readable outlines.
- `prompts/` – specialised prompt templates for scenario synthesis and automated grading (planned).
- `build_scenario.py` – helper to expand structured outlines into full JSON via Azure OpenAI.
- `reporting.py` – utilities to collate metrics outputs (planned).

As scripts mature they will mirror the workflow described in `docs/llm_validation_pipeline.md` but tailored for depth/breadth/span/complexity stress dimensions.
