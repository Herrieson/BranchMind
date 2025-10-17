import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def render_prompt(template: str, bundle: Dict[str, Dict]) -> str:
    return (
        f"{template.rstrip()}\n\n"
        "SCENARIO_JSON:\n"
        f"{json.dumps(bundle['scenario'], ensure_ascii=False, separators=(',', ':'))}\n\n"
        "TRANSCRIPT_JSON:\n"
        f"{json.dumps(bundle['transcript'], ensure_ascii=False, separators=(',', ':'))}\n\n"
        "INSTRUMENTATION_JSON:\n"
        f"{json.dumps(bundle['instrumentation'], ensure_ascii=False, separators=(',', ':'))}"
    )


def evaluate_assistant(
    evaluator: AzureLLM,
    template: str,
    scenario_payload: Dict,
    transcript_payload: Dict,
    instrumentation: Dict,
    assistant_label: str,
) -> Dict:
    bundle = {
        "scenario": scenario_payload,
        "transcript": sanitize_transcript(transcript_payload),
        "instrumentation": instrumentation,
    }
    prompt = render_prompt(template, bundle)
    messages = [
        {"role": "system", "content": "You are an impartial evaluator who must follow instructions exactly."},
        {"role": "user", "content": prompt},
    ]
    response = evaluator.chat(messages, json_mode=True)
    return json.loads(response)


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
        summary[label]["task_success_rate"] += overall.get("task_success_rate", 0.0)
        summary[label]["forgetfulness_rate"] += overall.get("forgetfulness_rate", 0.0)
        controller_rate = overall.get("controller_accuracy_rate")
        if controller_rate is not None:
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

    template = args.prompt_path.read_text(encoding="utf-8")
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
            unmet = detect_unmet_commitments(transcript_payload)
            result = evaluate_assistant(evaluator, template, scenario_payload, transcript_payload, instrumentation, assistant)
            if unmet:
                result.setdefault("auto_findings", []).extend(unmet)
            result_path = output_dir / f"{scenario_id}_{assistant}.json"
            dump_json(result_path, result)
            artifacts.append(result)

    summary = compute_summary(artifacts)
    dump_json(output_dir / "summary.json", {"summary": summary, "samples": len(artifacts)})
    print(f"[analysis] Completed evaluation for {len(artifacts)} assistant runs. Summary written to {output_dir / 'summary.json'}.")


if __name__ == "__main__":
    main()
