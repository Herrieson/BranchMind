# BranchMind Stress Scenario Metadata Synthesiser

You have the completed transcript outline for a BranchMind stress scenario. Derive
evaluation metadata that will guide automated scoring.

Scenario summary:
<<SCENARIO_SUMMARY>>

Thread catalogue:
<<THREAD_CATALOG>>

Transcript digest (user turns only, ordered oldest → newest, truncated where needed):
<<TRANSCRIPT_DIGEST>>

Respond **only** with JSON:
{
  "evaluation_rubric": [
    {
      "criterion": "<metric name>",
      "description": "<what success looks like>",
      "weight": <float between 0 and 1>
    }
  ],
  "notes": {
    "multi_turn_requirements": ["<requirement>", "..."],
    "expected_failure_modes": ["<likely failure>", "..."]
  }
}

Guidelines:
- Provide 3–5 rubric items whose weights sum to 1.0 (allow 0.99–1.01 rounding).
- Include coverage for memory fidelity, conflict resolution, and responsiveness under
  pressure.
- `multi_turn_requirements` should mention critical follow-ups or staged tasks that the
  assistant must revisit later in the scenario.
- `expected_failure_modes` should call out realistic mistakes (forgetting constraints,
  mixing threads, mishandling emergencies, etc.).
