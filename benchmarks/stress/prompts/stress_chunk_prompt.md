# BranchMind Stress Scenario Chunk Builder

You are writing the next block of user turns for a BranchMind stress scenario.
Follow the previously agreed scenario plan and extend the dialogue with realistic,
high-pressure user requests.

Scenario metadata (reference only):
<<SCENARIO_METADATA>>

Thread catalogue (do not invent new IDs):
<<THREAD_CATALOG>>

Chunk directive:
<<CHUNK_DIRECTIVE>>

Existing transcript excerpt (most recent turns first, truncate where necessary):
<<TRANSCRIPT_EXCERPT>>

Global requirements to respect:
<<GLOBAL_REQUIREMENTS>>

Instructions:
- Produce exactly `turn_target` user turns (unless fewer are needed to conclude the
  scenario, which will be explicitly flagged in the directive).
- Each turn must align with one `thread_id` from the catalogue.
- Interleave objectives and escalations as described.
- Keep the Chinese tone and detail level consistent with the outline (dates, budgets,
  conflicting constraints, etc.).
- Provide `expected_assistant_focus` summarising what a well-performing assistant should
  address for that user message.
- Use `timestamp_hint` or `escalation_trigger` only when materially helpful.

Respond **only** with JSON:
{
  "turns": [
    {
      "thread_id": "<thread from catalogue>",
      "user_message": "<full user utterance>",
      "expected_assistant_focus": "<assistant focus summary>",
      "timestamp_hint": "<optional, e.g. '周二下午'>",
      "escalation_trigger": "<optional label signalling priority>"
    }
  ],
  "summary": "<one sentence chunk summary for human trackers>"
}

Notes:
- Stick to user turns only; do not include assistant responses.
- Ensure chronology remains coherent across threads (keep calendars consistent).
- `summary` is for logging only and will not be stored in the final scenario.
