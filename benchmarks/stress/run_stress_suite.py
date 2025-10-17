import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.common import (
    GlobalConfig,
    LLMRetryConfig,
    Scenario,
    ScenarioTurn,
    TranscriptBundle,
    TranscriptTurn,
    build_azure_client,
    dump_json,
    load_scenarios,
    seed_everything,
)
from benchmarks.validation.run_evaluation import (
    SlidingWindowBaseline,
    build_branchmind_namespace,
    build_transcript_turn,
    create_baseline,
)
from benchmarks.validation.run_full_history_baseline import create_full_history_baseline
from branchmind import AppConfig, DialogueManager, DialogueObserver, LLMCallRecord, TreeSnapshot


ASSISTANT_BRANCHMIND = "branchmind"
ASSISTANT_SLIDING = "sliding_window"
ASSISTANT_FULL_HISTORY = "full_history"
SUPPORTED_ASSISTANTS = {ASSISTANT_BRANCHMIND, ASSISTANT_SLIDING, ASSISTANT_FULL_HISTORY}


def render_progress(label: str, current: int, total: int, width: int = 30) -> None:
    """Render a lightweight progress bar without external dependencies."""
    if total <= 0:
        return
    fraction = min(max(current / total, 0.0), 1.0)
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{label}] [{bar}] {current}/{total}", end="", flush=True)
    if current >= total:
        print()


def serialize_tree_decision(decision) -> Dict[str, Any]:
    return {
        "operation": decision.operation.value,
        "primary_branch": decision.primary_branch,
        "target_parent_branch": decision.target_parent_branch,
        "secondary_branches": list(decision.secondary_branches),
        "context_branches": list(decision.context_branches),
        "archive_targets": list(decision.archive_targets),
        "note": decision.note,
    }


def serialize_planned_operation(plan) -> Dict[str, Any]:
    return {
        "operation": plan.operation.value,
        "parent_id": plan.parent_id,
        "context_branch_ids": list(plan.context_branch_ids),
        "archive_targets": list(plan.archive_targets),
        "metadata": dict(plan.metadata),
    }


@dataclass
class TurnLog:
    round_index: int
    scenario_turn: Dict[str, Any]
    query: str
    latency: float
    assistant_response: str
    total_tokens: int = 0
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)
    controller_decision: Optional[Dict[str, Any]] = None
    planned_operation: Optional[Dict[str, Any]] = None
    tree_snapshot: Optional[Dict[str, Any]] = None
    node: Optional[Dict[str, Any]] = None


class StressObserver(DialogueObserver):
    def __init__(self) -> None:
        self.turns: List[TurnLog] = []
        self._current: Optional[TurnLog] = None

    def start_turn(self, *, round_index: int, turn: ScenarioTurn) -> None:
        if self._current is not None:
            raise RuntimeError("Previous turn not finalised before starting a new one.")
        self._current = TurnLog(
            round_index=round_index,
            scenario_turn=turn.to_dict(),
            query=turn.user_message,
            latency=0.0,
            assistant_response="",
        )

    def finish_turn(self, *, response: str, latency: float) -> None:
        if self._current is None:
            return
        self._current.assistant_response = response
        self._current.latency = latency
        self._current.total_tokens = sum(
            call.get("usage", {}).get("total_tokens", 0) for call in self._current.llm_calls
        )
        self.turns.append(self._current)
        self._current = None

    def on_llm_call(self, record: LLMCallRecord) -> None:
        if self._current is None:
            return
        self._current.llm_calls.append(asdict(record))

    def on_decision(self, decision, plan) -> None:
        if self._current is None:
            return
        self._current.controller_decision = serialize_tree_decision(decision)
        self._current.planned_operation = serialize_planned_operation(plan)

    def on_tree_update(self, snapshot: TreeSnapshot, node) -> None:
        if self._current is None:
            return
        self._current.tree_snapshot = asdict(snapshot)
        self._current.node = node.to_dict()

    def export(self) -> List[Dict[str, Any]]:
        return [asdict(turn) for turn in self.turns]


class TurnTelemetry:
    def __init__(self, round_index: int, turn: ScenarioTurn) -> None:
        self.round_index = round_index
        self.turn = turn
        self.events: List[Dict[str, Any]] = []
        self.response_latency: float = 0.0
        self.assistant_response: str = ""

    def emit(self, phase: str, payload: Dict[str, Any]) -> None:
        entry = {"phase": phase, **payload}
        self.events.append(entry)

    def finalise(self, response: str, latency: float) -> Dict[str, Any]:
        self.response_latency = latency
        self.assistant_response = response
        total_tokens = sum(event.get("usage", {}).get("total_tokens", 0) for event in self.events)
        return {
            "round_index": self.round_index,
            "scenario_turn": self.turn.to_dict(),
            "query": self.turn.user_message,
            "assistant_response": response,
            "latency": latency,
            "llm_calls": self.events,
            "total_tokens": total_tokens,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run high-pressure stress scenarios across assistants.")
    parser.add_argument("--scenarios", type=Path, required=True, help="Scenario JSON file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("stress_results"))
    parser.add_argument(
        "--assistants",
        nargs="+",
        choices=sorted(SUPPORTED_ASSISTANTS),
        default=[ASSISTANT_BRANCHMIND, ASSISTANT_SLIDING, ASSISTANT_FULL_HISTORY],
        help="Assistants to execute.",
    )
    parser.add_argument("--scenario-limit", type=int, default=None, help="Optional limit on number of scenarios.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for deterministic ordering.")

    # Baseline knobs
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--baseline-summarizer-model", default=None)
    parser.add_argument("--baseline-window-turns", type=int, default=6)
    parser.add_argument("--baseline-summary-trigger", type=int, default=4)
    parser.add_argument(
        "--baseline-system-prompt",
        default="You are a diligent assistant using only the recent conversation and provided summaries.",
    )
    parser.add_argument(
        "--full-history-system-prompt",
        default="You are a diligent assistant with access to the full dialogue history.",
    )

    # Azure overrides
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--azure-endpoint", default=None)
    parser.add_argument("--api-version", default=None)
    parser.add_argument("--branchmind-cm-model", default=None)
    parser.add_argument("--branchmind-task-model", default=None)
    parser.add_argument("--branchmind-summarizer-model", default=None)
    parser.add_argument("--retry-attempts", type=int, default=LLMRetryConfig().attempts)
    parser.add_argument("--retry-delay", type=float, default=LLMRetryConfig().delay)
    parser.add_argument("--sleep-between-assistants", type=float, default=0.0)
    parser.add_argument("--sleep-between-turns", type=float, default=0.0)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def aggregate_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"rounds": 0}
    total_latency = sum(item.get("latency", 0.0) for item in records)
    total_tokens = sum(item.get("total_tokens", 0) for item in records)
    max_tokens = max((item.get("total_tokens", 0) for item in records), default=0)
    avg_latency = total_latency / len(records)
    avg_tokens = total_tokens / len(records) if records else 0.0
    return {
        "rounds": len(records),
        "avg_latency": avg_latency,
        "avg_tokens": avg_tokens,
        "total_tokens": total_tokens,
        "peak_tokens": max_tokens,
    }


def aggregate_branchmind(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = aggregate_metrics(records)
    if not records:
        return base
    max_depth = max(
        (item.get("tree_snapshot", {}).get("max_depth", 0) for item in records if item.get("tree_snapshot")),
        default=0,
    )
    total_nodes = max(
        (item.get("tree_snapshot", {}).get("total_nodes", 0) for item in records if item.get("tree_snapshot")),
        default=0,
    )
    base.update({"max_tree_depth": max_depth, "final_node_count": total_nodes})
    return base


def build_branchmind_manager(args: argparse.Namespace, state_path: Path, observer: StressObserver) -> DialogueManager:
    namespace = build_branchmind_namespace(args, state_path)
    config = AppConfig.from_args(namespace)
    if state_path.exists():
        state_path.unlink()
    return DialogueManager(config, observer=observer)


def run_branchmind(
    scenario: Scenario,
    args: argparse.Namespace,
    scenario_dir: Path,
) -> Dict[str, Any]:
    assistant_dir = scenario_dir / ASSISTANT_BRANCHMIND
    ensure_dir(assistant_dir)
    state_path = assistant_dir / "state.json"
    observer = StressObserver()
    manager = build_branchmind_manager(args, state_path, observer)

    transcript = TranscriptBundle(
        assistant_label=ASSISTANT_BRANCHMIND,
        scenario_id=scenario.scenario_id,
        model_name=manager.config.task_model,
        system_description="BranchMind tree-structured dialogue manager",
        turns=[],
    )

    for round_index, turn in enumerate(scenario.dialogue, start=1):
        observer.start_turn(round_index=round_index, turn=turn)
        transcript.turns.append(build_transcript_turn("user", turn.user_message, turn))
        start = time.perf_counter()
        response = manager.handle_request(turn.user_message, thread_id=turn.thread_id)
        latency = time.perf_counter() - start
        observer.finish_turn(response=response, latency=latency)
        transcript.turns.append(
            TranscriptTurn(role="assistant", content=response, meta={"source": ASSISTANT_BRANCHMIND})
        )
        if args.sleep_between_turns:
            time.sleep(args.sleep_between_turns)

    manager.save_state()

    metrics: List[Dict[str, Any]] = observer.export()

    metrics_path = assistant_dir / "metrics.jsonl"
    write_jsonl(metrics_path, metrics)
    dump_json(assistant_dir / "transcript.json", transcript.to_dict())
    summary_path = assistant_dir / "summary.json"
    dump_json(summary_path, aggregate_branchmind(metrics))
    return {"metrics": metrics_path, "summary": summary_path, "state": state_path}


def run_sliding_baseline(
    scenario: Scenario,
    args: argparse.Namespace,
    scenario_dir: Path,
    client,
) -> Dict[str, Any]:
    assistant_dir = scenario_dir / ASSISTANT_SLIDING
    ensure_dir(assistant_dir)

    baseline = create_baseline(args, client)
    transcript = TranscriptBundle(
        assistant_label=ASSISTANT_SLIDING,
        scenario_id=scenario.scenario_id,
        model_name=baseline.llm.default_model,
        system_description="Sliding window baseline with periodic summarization",
        turns=[],
    )

    metrics: List[Dict[str, Any]] = []

    total_turns = len(scenario.dialogue)
    progress_label = f"{ASSISTANT_SLIDING}:{scenario.scenario_id}"
    if total_turns:
        render_progress(progress_label, 0, total_turns)

    for round_index, turn in enumerate(scenario.dialogue, start=1):
        recorder = TurnTelemetry(round_index, turn)
        transcript.turns.append(build_transcript_turn("user", turn.user_message, turn))
        start = time.perf_counter()
        response = baseline.handle_request(turn.user_message, telemetry=recorder.emit)
        latency = time.perf_counter() - start
        record = recorder.finalise(response, latency)
        transcript.turns.append(TranscriptTurn(role="assistant", content=response, meta={"source": ASSISTANT_SLIDING}))
        metrics.append(record)
        if total_turns:
            render_progress(progress_label, round_index, total_turns)
        if args.sleep_between_turns:
            time.sleep(args.sleep_between_turns)

    transcript_path = assistant_dir / "transcript.json"
    dump_json(transcript_path, transcript.to_dict())
    metrics_path = assistant_dir / "metrics.jsonl"
    summary_path = assistant_dir / "summary.json"
    write_jsonl(metrics_path, metrics)
    dump_json(summary_path, aggregate_metrics(metrics))
    return {"metrics": metrics_path, "summary": summary_path, "transcript": transcript_path}


def run_full_history_baseline(
    scenario: Scenario,
    args: argparse.Namespace,
    scenario_dir: Path,
    client,
) -> Dict[str, Any]:
    assistant_dir = scenario_dir / ASSISTANT_FULL_HISTORY
    ensure_dir(assistant_dir)
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.baseline_system_prompt = args.full_history_system_prompt
    baseline = create_full_history_baseline(baseline_args, client)
    transcript = TranscriptBundle(
        assistant_label=ASSISTANT_FULL_HISTORY,
        scenario_id=scenario.scenario_id,
        model_name=baseline.llm.default_model,
        system_description="Full history baseline",
        turns=[],
    )
    metrics: List[Dict[str, Any]] = []

    total_turns = len(scenario.dialogue)
    progress_label = f"{ASSISTANT_FULL_HISTORY}:{scenario.scenario_id}"
    if total_turns:
        render_progress(progress_label, 0, total_turns)

    for round_index, turn in enumerate(scenario.dialogue, start=1):
        recorder = TurnTelemetry(round_index, turn)
        transcript.turns.append(build_transcript_turn("user", turn.user_message, turn))
        start = time.perf_counter()
        response = baseline.handle_request(turn.user_message, telemetry=recorder.emit)
        latency = time.perf_counter() - start
        record = recorder.finalise(response, latency)
        transcript.turns.append(
            TranscriptTurn(role="assistant", content=response, meta={"source": ASSISTANT_FULL_HISTORY})
        )
        metrics.append(record)
        if total_turns:
            render_progress(progress_label, round_index, total_turns)
        if args.sleep_between_turns:
            time.sleep(args.sleep_between_turns)

    transcript_path = assistant_dir / "transcript.json"
    dump_json(transcript_path, transcript.to_dict())
    metrics_path = assistant_dir / "metrics.jsonl"
    summary_path = assistant_dir / "summary.json"
    write_jsonl(metrics_path, metrics)
    dump_json(summary_path, aggregate_metrics(metrics))
    return {"metrics": metrics_path, "summary": summary_path, "transcript": transcript_path}


def collect_scenarios(path: Path) -> List[Scenario]:
    if path.is_file():
        return load_scenarios(path)
    if not path.exists():
        raise FileNotFoundError(path)
    scenarios: List[Scenario] = []
    for json_file in sorted(path.glob("*.json")):
        scenarios.extend(load_scenarios(json_file))
    return scenarios


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    seed_everything(args.seed)
    GlobalConfig.env_overrides = {
        key: value
        for key, value in vars(args).items()
        if key in {"api_key", "azure_endpoint", "api_version"} and value
    }

    scenarios = collect_scenarios(args.scenarios)
    if args.scenario_limit:
        scenarios = scenarios[: args.scenario_limit]

    client = build_azure_client(args)

    manifest: List[Dict[str, Any]] = []

    for scenario in scenarios:
        scenario_dir = args.output_dir / scenario.scenario_id
        ensure_dir(scenario_dir)
        dump_json(scenario_dir / "scenario.json", scenario.to_dict())

        record: Dict[str, Any] = {"scenario_id": scenario.scenario_id, "title": scenario.title}

        if ASSISTANT_BRANCHMIND in args.assistants:
            record[ASSISTANT_BRANCHMIND] = run_branchmind(scenario, args, scenario_dir)
            if args.sleep_between_assistants:
                time.sleep(args.sleep_between_assistants)
        if ASSISTANT_SLIDING in args.assistants:
            record[ASSISTANT_SLIDING] = run_sliding_baseline(scenario, args, scenario_dir, client)
            if args.sleep_between_assistants:
                time.sleep(args.sleep_between_assistants)
        if ASSISTANT_FULL_HISTORY in args.assistants:
            record[ASSISTANT_FULL_HISTORY] = run_full_history_baseline(scenario, args, scenario_dir, client)
            if args.sleep_between_assistants:
                time.sleep(args.sleep_between_assistants)

        manifest.append(record)

    dump_json(args.output_dir / "manifest.json", manifest)
    print(f"[stress] Completed {len(manifest)} scenarios. Results stored in {args.output_dir}.")


if __name__ == "__main__":
    main()
