import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.common import (
    AzureLLM,
    GlobalConfig,
    LLMRetryConfig,
    build_azure_client,
    dump_json,
    load_env,
    seed_everything,
)
from benchmarks.stress.run_stress_suite import (
    ASSISTANT_BRANCHMIND,
    ASSISTANT_FULL_HISTORY,
    ASSISTANT_SLIDING,
    SUPPORTED_ASSISTANTS,
)


USER_CONTENT_CHAR_LIMIT = 500
ASSISTANT_CONTENT_CHAR_LIMIT = 900
EXPECTED_FOCUS_CHAR_LIMIT = 400
CONTROLLER_NOTE_CHAR_LIMIT = 300


@dataclass(frozen=True)
class EvaluationDimension:
    name: str
    rate_key: str
    per_turn_field: str
    prompt: str
    applies_to_branchmind_only: bool = False


TASK_SUCCESS_PROMPT = """
You assess task completion for long-form BranchMind stress scenarios.

You will receive scenario context, structural metrics, transcript excerpts, and instrumentation logs. For each user turn decide whether the assistant satisfied the stated or implied goal.

Return a JSON object exactly in the following shape:
{
  "dimension": "task_success",
  "per_turn": [
    {
      "round_index": 1,
      "turn_id": "U1",
      "task_success": true,
      "notes": "optional explanation <= 20 words"
    }
  ],
  "overall": {
    "task_success_rate": 0.0,
    "verdict": "<= 40 words synthesis",
    "strengths": ["bullet"],
    "risks": ["bullet"]
  }
}

Rules:
- Inspect only user turns; align round_index/turn_id with the transcript.
- Use true only when the assistant fully satisfies the goal, false when it fails, null when evidence is insufficient.
- Round task_success_rate to three decimals and keep it between 0 and 1.
- Keep notes short (<= 20 words) and omit or use empty string when unnecessary.
- Do not add extra keys or commentary outside the JSON object.
""".strip()


FORGETFULNESS_PROMPT = """
You audit context management for BranchMind stress scenarios.

Determine whether the assistant forgets, contradicts, or ignores previously established commitments or facts on each user turn.

Return JSON with this structure:
{
  "dimension": "forgetfulness",
  "per_turn": [
    {
      "round_index": 1,
      "turn_id": "U1",
      "forgetfulness": false,
      "notes": "optional explanation <= 20 words"
    }
  ],
  "overall": {
    "forgetfulness_rate": 0.0,
    "verdict": "<= 40 words synthesis",
    "strengths": ["bullet"],
    "risks": ["bullet"]
  }
}

Rules:
- Mark true when the assistant drops or contradicts salient context; false when it maintains context; null when you cannot tell.
- Rates must be floats between 0 and 1 rounded to three decimals.
- Keep notes short (<= 20 words) and omit when unnecessary.
- Output only the specified keys.
""".strip()


CONTROLLER_PROMPT = """
You judge BranchMind controller decisions for each user turn.

Use the transcript, controller telemetry, and scenario threads to decide whether the chosen tree operation matches the user's need. If the controller is missing for a turn, output null.

Return JSON exactly like:
{
  "dimension": "controller_accuracy",
  "per_turn": [
    {
      "round_index": 1,
      "turn_id": "U1",
      "controller_accuracy": true,
      "notes": "optional explanation <= 20 words"
    }
  ],
  "overall": {
    "controller_accuracy_rate": 0.0,
    "verdict": "<= 40 words synthesis",
    "strengths": ["bullet"],
    "risks": ["bullet"]
  }
}

Rules:
- Only evaluate BranchMind runs. For turns without controller metadata use null.
- Rates must be floats between 0 and 1 rounded to three decimals.
- Keep notes concise (<= 20 words). No extra keys or commentary.
""".strip()


DIMENSIONS: Sequence[EvaluationDimension] = (
    EvaluationDimension(
        name="task_success",
        rate_key="task_success_rate",
        per_turn_field="task_success",
        prompt=TASK_SUCCESS_PROMPT,
    ),
    EvaluationDimension(
        name="forgetfulness",
        rate_key="forgetfulness_rate",
        per_turn_field="forgetfulness",
        prompt=FORGETFULNESS_PROMPT,
    ),
    EvaluationDimension(
        name="controller_accuracy",
        rate_key="controller_accuracy_rate",
        per_turn_field="controller_accuracy",
        prompt=CONTROLLER_PROMPT,
        applies_to_branchmind_only=True,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated quality analysis for stress-test runs.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with stress suite outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store analysis artifacts (defaults to <results-dir>/analysis).",
    )
    parser.add_argument(
        "--assistants",
        nargs="+",
        choices=sorted(SUPPORTED_ASSISTANTS),
        default=sorted(SUPPORTED_ASSISTANTS),
        help="Assistant labels to evaluate.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("benchmarks/stress/prompts/stress_evaluator_prompt.md"),
        help="Prompt template for the evaluator model.",
    )
    parser.add_argument("--model", default=None, help="Override evaluator model (defaults to EVALUATOR_MODEL/TASK_MODEL).")
    parser.add_argument("--retry-attempts", type=int, default=LLMRetryConfig().attempts)
    parser.add_argument("--retry-delay", type=float, default=LLMRetryConfig().delay)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument(
        "--evaluator-replicas",
        type=int,
        default=3,
        help="Number of evaluator passes to run per dimension (>=1).",
    )
    parser.add_argument(
        "--enable-commitment-heuristics",
        action="store_true",
        help="Add optional rule-based diagnostics for calendar merge commitments.",
    )
    return parser.parse_args()


def collect_scenario_dirs(results_dir: Path) -> List[Path]:
    scenario_dirs: List[Path] = []
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and (path / "scenario.json").exists():
            scenario_dirs.append(path)
    return scenario_dirs


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _content_limit_for_role(role: Optional[str]) -> Optional[int]:
    if role == "assistant":
        return ASSISTANT_CONTENT_CHAR_LIMIT
    if role == "user":
        return USER_CONTENT_CHAR_LIMIT
    return None


def _truncate_content(text: str, limit: Optional[int]) -> str:
    if not limit or len(text) <= limit:
        return text
    suffix = f"...[TRUNCATED {len(text) - limit} chars]"
    truncated = text[:limit].rstrip()
    if truncated.endswith("..."):
        return truncated + suffix
    return truncated + suffix


def sanitize_transcript(transcript_payload: Dict) -> Dict:
    sanitized: Dict[str, Any] = {
        key: value
        for key, value in transcript_payload.items()
        if key != "turns"
    }
    turns: List[Dict[str, Any]] = []
    for turn in transcript_payload.get("turns", []):
        trimmed = {k: v for k, v in turn.items() if k != "meta"}
        role = trimmed.get("role")
        if "content" in trimmed and isinstance(trimmed["content"], str):
            limit = _content_limit_for_role(role)
            trimmed["content"] = _truncate_content(trimmed["content"], limit)
        turns.append(trimmed)
    sanitized["turns"] = turns
    return sanitized


def _sanitize_metric_record(payload: Dict, assistant_label: str) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "round_index": payload.get("round_index"),
    }
    scenario_turn = payload.get("scenario_turn") or {}
    scenario_snapshot: Dict[str, Any] = {}
    for key in ("turn_id", "thread_id", "expected_assistant_focus", "escalation_trigger"):
        value = scenario_turn.get(key)
        if value is not None:
            if key == "expected_assistant_focus" and isinstance(value, str):
                scenario_snapshot[key] = _truncate_content(value, EXPECTED_FOCUS_CHAR_LIMIT)
            else:
                scenario_snapshot[key] = value
    if scenario_snapshot:
        record["scenario_turn"] = scenario_snapshot

    if assistant_label == ASSISTANT_BRANCHMIND:
        controller_decision = payload.get("controller_decision")
        if controller_decision:
            decision_snapshot: Dict[str, Any] = {}
            for key in (
                "operation",
                "primary_branch",
                "target_parent_branch",
                "secondary_branches",
                "context_branches",
                "archive_targets",
            ):
                value = controller_decision.get(key)
                if value:
                    decision_snapshot[key] = value
            note = controller_decision.get("note")
            if isinstance(note, str):
                decision_snapshot["note"] = _truncate_content(note, CONTROLLER_NOTE_CHAR_LIMIT)
            elif note:
                decision_snapshot["note"] = note
            if decision_snapshot:
                record["controller_decision"] = decision_snapshot
        planned_operation = payload.get("planned_operation")
        if planned_operation:
            record["planned_operation"] = {
                key: planned_operation.get(key)
                for key in ("operation", "parent_id", "context_branch_ids", "archive_targets")
                if planned_operation.get(key)
            }
        tree_snapshot = payload.get("tree_snapshot") or {}
        if tree_snapshot:
            record["tree_snapshot"] = {
                key: tree_snapshot.get(key)
                for key in ("total_nodes", "active_nodes", "active_branches", "max_depth", "last_operation")
                if tree_snapshot.get(key) is not None
            }
    else:
        latency = payload.get("latency")
        if latency is not None:
            record["latency"] = latency
        total_tokens = payload.get("total_tokens")
        if total_tokens is not None:
            record["total_tokens"] = total_tokens
    return record


def load_metrics(path: Path, assistant_label: str) -> Dict:
    if not path.exists():
        return {}
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            payload.pop("assistant_response", None)
            if assistant_label == ASSISTANT_BRANCHMIND:
                payload.pop("node", None)
            records.append(_sanitize_metric_record(payload, assistant_label))
    return {"per_turn": records}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quantile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def derive_structural_metrics(
    transcript_payload: Dict[str, Any],
    instrumentation: Dict[str, Any],
    assistant_label: str,
) -> Dict[str, Any]:
    turns = transcript_payload.get("turns", []) or []
    user_turns = [turn for turn in turns if turn.get("role") == "user"]
    assistant_turns = [turn for turn in turns if turn.get("role") == "assistant"]

    summary = instrumentation.get("summary") or {}
    per_turn = instrumentation.get("per_turn") or []

    latencies = [float(item["latency"]) for item in per_turn if _is_number(item.get("latency"))]
    tokens = [float(item["total_tokens"]) for item in per_turn if _is_number(item.get("total_tokens"))]

    latency_stats: Dict[str, float] = {}
    if latencies:
        latency_stats = {
            "avg": sum(latencies) / len(latencies),
            "min": min(latencies),
            "max": max(latencies),
        }
        p90 = _quantile(latencies, 0.9)
        if p90 is not None:
            latency_stats["p90"] = p90

    token_stats: Dict[str, float] = {}
    if tokens:
        token_stats = {
            "avg": sum(tokens) / len(tokens),
            "min": min(tokens),
            "max": max(tokens),
        }
        p90_tokens = _quantile(tokens, 0.9)
        if p90_tokens is not None:
            token_stats["p90"] = p90_tokens

    flags: List[str] = []
    avg_latency = summary.get("avg_latency") or latency_stats.get("avg")
    if _is_number(avg_latency) and avg_latency > 20.0:
        flags.append("avg_latency_high")
    peak_tokens = summary.get("peak_tokens") or token_stats.get("max")
    if _is_number(peak_tokens) and peak_tokens > 8000:
        flags.append("peak_tokens_high")
    if len(user_turns) >= 120:
        flags.append("long_dialogue")

    branchmind_stats: Dict[str, Any] = {}
    if assistant_label == ASSISTANT_BRANCHMIND:
        tree_depths = []
        node_counts = []
        for record in per_turn:
            snapshot = record.get("tree_snapshot") or {}
            depth = snapshot.get("max_depth")
            if _is_number(depth):
                tree_depths.append(float(depth))
            nodes = snapshot.get("total_nodes")
            if _is_number(nodes):
                node_counts.append(float(nodes))
        if tree_depths:
            branchmind_stats["max_tree_depth_observed"] = max(tree_depths)
            if max(tree_depths) >= 20:
                flags.append("tree_depth_spike")
        if node_counts:
            branchmind_stats["max_node_count_observed"] = max(node_counts)

    return {
        "counts": {
            "conversation_turns": len(turns),
            "user_turns": len(user_turns),
            "assistant_turns": len(assistant_turns),
        },
        "summary": summary,
        "latency": latency_stats,
        "token_usage": token_stats,
        "branchmind": branchmind_stats,
        "flags": flags,
    }
def render_dimension_prompt(
    dimension: EvaluationDimension,
    bundle: Dict[str, Dict[str, Any]],
    extra_guidance: Optional[str] = None,
) -> str:
    sections = [dimension.prompt]
    if extra_guidance:
        sections.append(f"ADDITIONAL_GUIDANCE:\n{extra_guidance.strip()}")
    sections.append(
        "SCENARIO_JSON:\n"
        f"{json.dumps(bundle['scenario'], ensure_ascii=False, separators=(',', ':'))}"
    )
    sections.append(
        "STRUCTURAL_METRICS_JSON:\n"
        f"{json.dumps(bundle['structural'], ensure_ascii=False, separators=(',', ':'))}"
    )
    sections.append(
        "TRANSCRIPT_JSON:\n"
        f"{json.dumps(bundle['transcript'], ensure_ascii=False, separators=(',', ':'))}"
    )
    sections.append(
        "INSTRUMENTATION_JSON:\n"
        f"{json.dumps(bundle['instrumentation'], ensure_ascii=False, separators=(',', ':'))}"
    )
    return "\n\n".join(sections)


def _validate_dimension_payload(dimension: EvaluationDimension, payload: Dict[str, Any]) -> None:
    per_turn = payload.get("per_turn")
    if not isinstance(per_turn, list):
        raise ValueError(f"{dimension.name}: 'per_turn' must be a list.")
    for entry in per_turn:
        if not isinstance(entry, dict):
            raise ValueError(f"{dimension.name}: per_turn entries must be objects.")
        if dimension.per_turn_field not in entry:
            raise ValueError(f"{dimension.name}: missing '{dimension.per_turn_field}' in per_turn entry.")
    overall = payload.get("overall")
    if not isinstance(overall, dict):
        raise ValueError(f"{dimension.name}: 'overall' must be an object.")
    rate = overall.get(dimension.rate_key)
    if rate is not None and not _is_number(rate):
        raise ValueError(f"{dimension.name}: '{dimension.rate_key}' must be numeric or null.")


def evaluate_dimension(
    evaluator: AzureLLM,
    dimension: EvaluationDimension,
    bundle: Dict[str, Dict[str, Any]],
    extra_guidance: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = render_dimension_prompt(dimension, bundle, extra_guidance)
    messages = [
        {"role": "system", "content": "You are an impartial evaluator who must follow instructions exactly and output JSON only."},
        {"role": "user", "content": prompt},
    ]
    response = evaluator.chat(messages, json_mode=True)
    payload = json.loads(response)
    if not isinstance(payload, dict):
        raise ValueError(f"{dimension.name}: evaluator returned non-object payload.")
    payload["dimension"] = dimension.name
    _validate_dimension_payload(dimension, payload)
    return payload


DIMENSION_MAP = {dimension.name: dimension for dimension in DIMENSIONS}


def _majority_vote(values: Sequence[Optional[bool]]) -> Optional[bool]:
    bool_values = [value for value in values if isinstance(value, bool)]
    if not bool_values:
        return None
    true_count = sum(1 for value in bool_values if value)
    false_count = len(bool_values) - true_count
    if true_count > false_count:
        return True
    if false_count > true_count:
        return False
    return None


def aggregate_dimension_results(dimension: EvaluationDimension, results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        raise ValueError(f"No evaluation results provided for dimension '{dimension.name}'.")
    if len(results) == 1:
        return results[0]

    per_turn_order: List[Tuple[Any, Any]] = []
    per_turn_meta: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    per_turn_votes: Dict[Tuple[Any, Any], List[Optional[bool]]] = {}
    per_turn_notes: Dict[Tuple[Any, Any], List[str]] = {}

    for result in results:
        for entry in result.get("per_turn", []):
            round_index = entry.get("round_index")
            turn_id = entry.get("turn_id")
            key = (round_index, turn_id)
            if key not in per_turn_meta:
                per_turn_meta[key] = {k: v for k, v in entry.items() if k in {"round_index", "turn_id"} and v is not None}
                per_turn_order.append(key)
            per_turn_votes.setdefault(key, []).append(entry.get(dimension.per_turn_field))
            note = entry.get("notes")
            if isinstance(note, str) and note.strip():
                per_turn_notes.setdefault(key, []).append(note.strip())

    aggregated_per_turn: List[Dict[str, Any]] = []
    for key in per_turn_order:
        entry_meta = dict(per_turn_meta.get(key, {}))
        votes = per_turn_votes.get(key, [])
        entry_meta[dimension.per_turn_field] = _majority_vote(votes)
        notes = per_turn_notes.get(key, [])
        if notes:
            entry_meta["notes"] = "; ".join(dict.fromkeys(notes))
        aggregated_per_turn.append(entry_meta)

    overall_components = [result.get("overall", {}) for result in results]
    base_overall: Dict[str, Any] = {}
    if overall_components:
        keys = set().union(*(component.keys() for component in overall_components))
        for key in keys:
            if key == dimension.rate_key:
                numeric_values = [component.get(key) for component in overall_components if _is_number(component.get(key))]
                if numeric_values:
                    base_overall[key] = round(sum(numeric_values) / len(numeric_values), 3)
                else:
                    base_overall[key] = None
            elif key in {"strengths", "risks"}:
                items: List[str] = []
                for component in overall_components:
                    values = component.get(key)
                    if isinstance(values, list):
                        items.extend(str(item) for item in values if str(item))
                base_overall[key] = list(dict.fromkeys(items))
            elif key == "verdict":
                verdicts = [
                    str(component.get("verdict")).strip()
                    for component in overall_components
                    if isinstance(component.get("verdict"), str) and component.get("verdict").strip()
                ]
                base_overall[key] = " | ".join(dict.fromkeys(verdicts))
            else:
                # Preserve other scalar keys from the first component
                for component in overall_components:
                    value = component.get(key)
                    if value is not None:
                        base_overall[key] = value
                        break
        base_overall.setdefault(dimension.rate_key, None)
        base_overall.setdefault("strengths", [])
        base_overall.setdefault("risks", [])
        base_overall.setdefault("verdict", "")

    return {
        "dimension": dimension.name,
        "per_turn": aggregated_per_turn,
        "overall": base_overall,
        "replicas": len(results),
    }


def evaluate_dimensions(
    evaluator: AzureLLM,
    scenario_payload: Dict[str, Any],
    transcript_payload: Dict[str, Any],
    instrumentation: Dict[str, Any],
    structural: Dict[str, Any],
    assistant_label: str,
    extra_guidance: Optional[str] = None,
    replicas: int = 1,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    bundle = {
        "scenario": scenario_payload,
        "transcript": transcript_payload,
        "instrumentation": instrumentation,
        "structural": structural,
    }
    replicas = max(1, int(replicas))
    raw_results: Dict[str, List[Dict[str, Any]]] = {dimension.name: [] for dimension in DIMENSIONS}
    for iteration in range(replicas):
        for dimension in DIMENSIONS:
            if dimension.applies_to_branchmind_only and assistant_label != ASSISTANT_BRANCHMIND:
                continue
            result = evaluate_dimension(evaluator, dimension, bundle, extra_guidance)
            raw_results.setdefault(dimension.name, []).append(result)

    aggregated: Dict[str, Dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        if dimension.applies_to_branchmind_only and assistant_label != ASSISTANT_BRANCHMIND:
            continue
        aggregated[dimension.name] = aggregate_dimension_results(dimension, raw_results.get(dimension.name, []))
    return aggregated, raw_results


def assemble_overall(assistant_label: str, dimension_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    overall: Dict[str, Any] = {
        "task_success_rate": None,
        "forgetfulness_rate": None,
        "controller_accuracy_rate": None,
        "strengths": [],
        "risks": [],
        "verdict": "",
    }
    verdict_parts: List[str] = []

    for name, result in dimension_results.items():
        dimension = DIMENSION_MAP[name]
        dim_overall = result.get("overall", {})
        rate = dim_overall.get(dimension.rate_key)
        if _is_number(rate):
            overall[dimension.rate_key] = float(rate)
        elif rate is None and overall.get(dimension.rate_key) is None:
            overall[dimension.rate_key] = None

        strengths = dim_overall.get("strengths", [])
        if isinstance(strengths, list):
            overall["strengths"].extend(str(item) for item in strengths if str(item))

        risks = dim_overall.get("risks", [])
        if isinstance(risks, list):
            overall["risks"].extend(str(item) for item in risks if str(item))

        verdict = dim_overall.get("verdict")
        if isinstance(verdict, str) and verdict.strip():
            verdict_parts.append(f"{name}: {verdict.strip()}")

    if assistant_label != ASSISTANT_BRANCHMIND:
        overall["controller_accuracy_rate"] = None

    # Deduplicate while preserving order
    overall["strengths"] = list(dict.fromkeys(overall["strengths"]))
    overall["risks"] = list(dict.fromkeys(overall["risks"]))
    overall["verdict"] = " ".join(verdict_parts).strip()

    return overall


def detect_unmet_commitments(transcript_payload: Dict) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = transcript_payload.get("turns", [])
    findings: List[Dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if turn.get("role") != "user":
            continue
        content = str(turn.get("content", ""))
        lowered = content.lower()
        if not any(keyword in lowered for keyword in ["merge", "合并"]) and "ics" not in lowered:
            continue
        assistant_reply = ""
        if index + 1 < len(turns) and turns[index + 1].get("role") == "assistant":
            assistant_reply = str(turns[index + 1].get("content", ""))
        reply_lower = assistant_reply.lower()
        missing: List[str] = []
        if "ics" not in reply_lower:
            missing.append("缺少 ICS 草案")
        if not any(keyword in reply_lower for keyword in ["冲突", "conflict"]):
            missing.append("缺少冲突检查")
        if not any(keyword in reply_lower for keyword in ["日历", "calendar", "时间线"]):
            missing.append("缺少主日历时间线")
        if missing:
            findings.append(
                {
                    "turn_index": index,
                    "issues": missing,
                    "user_request": content,
                    "assistant_reply_excerpt": assistant_reply[:400],
                }
            )
    return findings


def compute_summary(artifacts: List[Dict]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    controller_counts: Dict[str, int] = {}
    controller_sums: Dict[str, float] = {}
    for artifact in artifacts:
        label = artifact["assistant_label"]
        overall = artifact.get("overall", {})
        summary.setdefault(label, {"task_success_rate": 0.0, "forgetfulness_rate": 0.0, "controller_accuracy_rate": 0.0})
        counts[label] = counts.get(label, 0) + 1
        task_rate = overall.get("task_success_rate")
        if _is_number(task_rate):
            summary[label]["task_success_rate"] += float(task_rate)
        forgetfulness_rate = overall.get("forgetfulness_rate")
        if _is_number(forgetfulness_rate):
            summary[label]["forgetfulness_rate"] += float(forgetfulness_rate)
        controller_rate = overall.get("controller_accuracy_rate")
        if _is_number(controller_rate):
            controller_sums[label] = controller_sums.get(label, 0.0) + controller_rate
            controller_counts[label] = controller_counts.get(label, 0) + 1
    for label, aggregate in summary.items():
        count = counts[label]
        aggregate["task_success_rate"] = round(aggregate["task_success_rate"] / count, 4)
        aggregate["forgetfulness_rate"] = round(aggregate["forgetfulness_rate"] / count, 4)
        if controller_counts.get(label):
            aggregate["controller_accuracy_rate"] = round(
                controller_sums[label] / controller_counts[label], 4
            )
        else:
            aggregate["controller_accuracy_rate"] = None
        aggregate["samples"] = count
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.results_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    GlobalConfig.env_overrides = {
        key: value
        for key, value in vars(args).items()
        if key in {"api_key", "azure_endpoint", "api_version"} and value
    }

    guidance = args.prompt_path.read_text(encoding="utf-8").strip() if args.prompt_path.exists() else None
    client = build_azure_client(args)
    model_name = (
        args.model
        or load_env("STRESS_EVALUATOR_MODEL")
        or load_env("EVALUATOR_MODEL")
        or load_env("TASK_MODEL")
        or "gpt-4o"
    )
    evaluator = AzureLLM(client, model_name, retry=LLMRetryConfig(args.retry_attempts, args.retry_delay))

    scenario_dirs = collect_scenario_dirs(args.results_dir)
    artifacts: List[Dict] = []
    for scenario_dir in scenario_dirs:
        scenario_payload = load_json(scenario_dir / "scenario.json")
        scenario_id = scenario_payload.get("scenario_id") or scenario_dir.name
        for assistant in args.assistants:
            assistant_dir = scenario_dir / assistant
            if not assistant_dir.exists():
                continue
            transcript_path = assistant_dir / "transcript.json"
            if not transcript_path.exists():
                continue
            transcript_payload = load_json(transcript_path)
            instrumentation = {
                "summary": load_json(assistant_dir / "summary.json") if (assistant_dir / "summary.json").exists() else {},
                **load_metrics(assistant_dir / "metrics.jsonl", assistant),
            }

            print(f"[analysis] Evaluating {scenario_id} :: {assistant} with model '{model_name}'.")
            sanitized_transcript = sanitize_transcript(transcript_payload)
            structural = derive_structural_metrics(sanitized_transcript, instrumentation, assistant)
            aggregated_dimensions, raw_dimension_results = evaluate_dimensions(
                evaluator,
                scenario_payload,
                sanitized_transcript,
                instrumentation,
                structural,
                assistant,
                guidance,
                args.evaluator_replicas,
            )
            overall = assemble_overall(assistant, aggregated_dimensions)

            artifact: Dict[str, Any] = {
                "assistant_label": assistant,
                "scenario_id": scenario_id,
                "structural_analysis": structural,
                "dimensions": aggregated_dimensions,
                "overall": overall,
                "evaluation_metadata": {
                    "model": model_name,
                    "replicas": args.evaluator_replicas,
                },
            }
            if args.evaluator_replicas > 1:
                artifact["evaluation_metadata"]["raw_dimension_votes"] = {
                    name: len(outputs) for name, outputs in raw_dimension_results.items() if outputs
                }

            diagnostics: Dict[str, Any] = {}
            structural_flags = structural.get("flags") or []
            if structural_flags:
                diagnostics["structural_flags"] = structural_flags
            if args.enable_commitment_heuristics:
                commitment_findings = detect_unmet_commitments(transcript_payload)
                if commitment_findings:
                    diagnostics["commitment_findings"] = commitment_findings
            if diagnostics:
                artifact["diagnostics"] = diagnostics

            result_path = output_dir / f"{scenario_id}_{assistant}.json"
            dump_json(result_path, artifact)
            artifacts.append(artifact)

    summary = compute_summary(artifacts)
    dump_json(output_dir / "summary.json", {"summary": summary, "samples": len(artifacts)})
    print(f"[analysis] Completed evaluation for {len(artifacts)} assistant runs. Summary written to {output_dir / 'summary.json'}.")


if __name__ == "__main__":
    main()
