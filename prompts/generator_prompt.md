# BranchMind Scenario Generator Prompt

You are helping us stress-test a dialogue manager named **BranchMind**. We need compact but challenging conversation scripts that force an assistant to juggle several concurrent topics (“threads”) and revisit them later.

Follow these rules **strictly** and respond with a single JSON object that matches the schema below.

```json
{
  "scenarios": [
    {
      "scenario_id": "S-###",
      "title": "Short human-readable title",
      "background": "What is happening? Include stakes, timelines, constraints.",
      "user_profile": "Persona of the user driving the dialogue.",
      "assistant_profile": "How the assistant is expected to behave (e.g., role, style).",
      "threads": [
        {
          "thread_id": "slug",
          "intent": "One-sentence objective for this thread.",
          "priority": "high|medium|low",
          "key_entities": ["entity/person/tool"]
        }
      ],
      "dialogue": [
        {
          "turn_id": "U1",
          "thread_id": "which thread is active (or 'mixed' if it blends two threads)",
          "timestamp_hint": "Optional diegetic timing like 'Day 2 - morning'.",
          "user_message": "Full natural-language utterance from the user.",
          "expected_assistant_focus": "What the assistant must keep track of or produce.",
          "escalation_trigger": "null or a short note describing what makes this turn difficult."
        }
      ],
      "evaluation_rubric": [
        {
          "criterion": "Context tracking / Dealing with interruptions / ...",
          "description": "How success looks for this criterion.",
          "weight": 0.0
        }
      ],
      "notes": {
        "multi_turn_requirements": [
          "Specific callbacks (e.g., 'Turn U5 must reference decisions from U2 and U3')."
        ],
        "expected_failure_modes": [
          "Concrete mistakes weaker systems will make."
        ]
      }
    }
  ]
}
```

Additional instructions:

1. Generate between **5 and 10** scenarios.
2. Vary domains (technical troubleshooting, project management, personal decisions, research synthesis, etc.).
3. Ensure at least two threads per scenario, with explicit crossing or revival of earlier threads.
4. Populate `expected_assistant_focus` with actionable guidance that will later help the evaluator judge responses.
5. Use fluent, idiomatic language appropriate to the background (can mix English and Chinese if natural for the scenario, but prefer one primary language per scenario).
6. Keep `scenario_id` unique and sequential (e.g., `S-101`, `S-102`, …).
7. Do **not** include assistant utterances; only user turns plus metadata.
8. Ensure the `weight` values are floats in the range `[0, 1]` that sum to exactly `1.0` for each `evaluation_rubric`.

Return only the JSON object—no commentary or Markdown.
