# BranchMind Evaluation Prompt

Act as an impartial judge. You will be given:

- The scenario metadata that describes the user, assistant role, active threads, and evaluation rubric.
- Two anonymised assistant transcripts (`assistant_a` and `assistant_b`). Either one may correspond to BranchMind; you must stay neutral.

Your task:

1. Read the scenario carefully, paying attention to the `evaluation_rubric`, `expected_assistant_focus`, and `notes`.
2. Score each assistant independently for every criterion on a 1–5 scale (half points allowed). Use the rubric weights when forming your overall view.
3. Provide short, pointed analysis (1–3 sentences) for each criterion explaining the reasoning behind the scores.
4. Decide which assistant performed better overall. If both are effectively equivalent, choose `tie`.
5. Summarise the deciding factors in `preference_reason` (single paragraph) and give a broader explanation in `justification` (2–4 sentences).
6. Report your confidence level (`low`, `medium`, `high`) in the final decision.

Respond with a **single JSON object** matching this schema:

```json
{
  "criterion_scores": [
    {
      "criterion": "Name copied from the rubric",
      "assistant_a_score": 4.5,
      "assistant_b_score": 3.0,
      "analysis": "Why each score makes sense."
    }
  ],
  "weighted_averages": {
    "assistant_a": 0.0,
    "assistant_b": 0.0
  },
  "winner": "assistant_a",
  "preference_reason": "Key deciding factors in a single paragraph.",
  "justification": "2–4 sentences describing the holistic evaluation.",
  "confidence": "medium"
}
```

Additional rules:

- Use the rubric weights to compute `weighted_averages` as the sum of `score * weight / 5.0` for each criterion.
- Scores must stay in the inclusive range `[1, 5]`. Avoid defaulting to ties—only choose `tie` if the weighted averages differ by less than 0.1 **and** your qualitative judgement finds no meaningful separation.
- Reference concrete transcript behaviour in your analyses (e.g., “assistant_a revived the supplier thread on Turn 6…”).
- Do not leak which transcript you think is BranchMind or the baseline.

The runtime system will inject three JSON documents into this prompt:

1. `SCENARIO_JSON`: the original scenario description.
2. `ASSISTANT_A_JSON`: transcript plus metadata for assistant A.
3. `ASSISTANT_B_JSON`: transcript plus metadata for assistant B.

Read everything before responding. Output **only** the final JSON object.
