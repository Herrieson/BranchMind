import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.common import (
    AzureLLM,
    GlobalConfig,
    LLMRetryConfig,
    Scenario,
    ScenarioTurn,
    TranscriptBundle,
    TranscriptTurn,
    build_azure_client,
    dump_json,
    load_env,
    load_scenarios,
    seed_everything,
)

from branchmind import (
    AppConfig,
    DialogueManager,
    DEFAULT_API_VERSION,
    DEFAULT_CM_MODEL,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_STATE_FILE,
    DEFAULT_SUMMARIZER_MODEL,
    DEFAULT_TASK_MODEL,
)


class SlidingWindowBaseline:
    """Baseline dialogue manager that keeps a sliding window plus running summary."""

    def __init__(
        self,
        llm: AzureLLM,
        summarizer: AzureLLM,
        *,
        window_turns: int,
        summary_trigger: int,
        system_prompt: str,
    ):
        self.llm = llm
        self.summarizer = summarizer
        self.window_turns = max(1, window_turns)
        self.summary_trigger = max(0, summary_trigger)
        self.system_prompt = system_prompt.strip() or "You are a helpful assistant."
        self.dialogue: List[Dict[str, str]] = []
        self.running_summaries: List[str] = []
        self.summary_cursor = 0

    @property
    def window_message_count(self) -> int:
        return self.window_turns * 2

    def handle_request(
        self,
        query: str,
        *,
        telemetry: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        def emit(label: str, payload: Dict[str, Any]) -> None:
            if telemetry:
                telemetry(label, payload)

        messages = self._build_prompt(query)
        response = self.llm.chat(messages, telemetry=lambda data: emit("main", data))
        self.dialogue.append({"role": "user", "content": query})
        self.dialogue.append({"role": "assistant", "content": response})
        self._maybe_summarize(telemetry=telemetry)
        return response

    def _build_prompt(self, user_query: str) -> List[Dict[str, str]]:
        window_messages = self.dialogue[-self.window_message_count :]
        system_messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if self.running_summaries:
            joined_summary = "\n".join(self.running_summaries[-3:])
            system_messages.append(
                {
                    "role": "system",
                    "content": f"[Running Summary]\n{joined_summary}",
                }
            )
        messages: List[Dict[str, str]] = system_messages + list(window_messages)
        messages.append({"role": "user", "content": user_query})
        return messages

    def _maybe_summarize(
        self,
        telemetry: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        if self.summary_trigger <= 0:
            return
        total_pairs = len(self.dialogue) // 2
        if total_pairs < self.summary_trigger:
            return
        cutoff = max(0, len(self.dialogue) - self.window_message_count)
        if cutoff <= self.summary_cursor:
            return
        segment = self.dialogue[self.summary_cursor : cutoff]
        if not segment:
            return
        segment_text = "\n".join(f"{item['role']}: {item['content']}" for item in segment)
        prompt = (
            "Summarize the following dialogue segment in <=120 words. Capture commitments, "
            "named entities, unresolved questions, and references that will matter later.\n\n"
            f"{segment_text}"
        )
        def emit(label: str, payload: Dict[str, Any]) -> None:
            if telemetry:
                telemetry(label, payload)

        summary = self.summarizer.chat(
            [
                {"role": "system", "content": "You are a concise dialogue summarizer."},
                {"role": "user", "content": prompt},
            ],
            telemetry=lambda data: emit("summary", data),
        )
        cleaned = summary.strip().replace("\n", " ")
        if cleaned:
            self.running_summaries.append(cleaned)
            self.summary_cursor = cutoff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute BranchMind vs Baseline on scenario scripts.")
    parser.add_argument(
        "--scenarios",
        type=Path,
        required=True,
        help="Path to a scenarios JSON file or a directory containing multiple scenario JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to store transcripts and intermediate artifacts.",
    )
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=None,
        help="Optionally limit the number of scenarios executed (useful for smoke tests).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios that already have both transcripts in the output directory.",
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help="Override baseline assistant model name (defaults to BASELINE_MODEL or TASK_MODEL).",
    )
    parser.add_argument(
        "--baseline-summarizer-model",
        default=None,
        help="Override baseline summarizer model (defaults to BASELINE_SUMMARIZER or SUMMARIZER_MODEL).",
    )
    parser.add_argument(
        "--baseline-window-turns",
        type=int,
        default=4,
        help="User-assistant turn pairs retained in the sliding window for the baseline model.",
    )
    parser.add_argument(
        "--baseline-summary-trigger",
        type=int,
        default=3,
        help="Summarize once this many user-assistant pairs accumulate outside the sliding window.",
    )
    parser.add_argument(
        "--baseline-system-prompt",
        default="You are a diligent assistant using only the recent conversation and provided summaries.",
        help="System prompt for the baseline assistant.",
    )
    parser.add_argument(
        "--branchmind-cm-model",
        default=None,
        help="Override BranchMind controller model (defaults to CM_MODEL).",
    )
    parser.add_argument(
        "--branchmind-task-model",
        default=None,
        help="Override BranchMind task model (defaults to TASK_MODEL).",
    )
    parser.add_argument(
        "--branchmind-summarizer-model",
        default=None,
        help="Override BranchMind summarizer model (defaults to SUMMARIZER_MODEL).",
    )
    parser.add_argument(
        "--branchmind-state-root",
        type=Path,
        default=None,
        help="Directory to store BranchMind state files (defaults to <output-dir>/branchmind_states).",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Retry attempts for both managers.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_DELAY,
        help="Delay between retries (seconds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (used for deterministic resume ordering).",
    )
    parser.add_argument("--api-key", default=None, help="Azure OpenAI API key override.")
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint override.")
    parser.add_argument("--api-version", default=None, help="Azure OpenAI API version override.")
    return parser.parse_args()


def collect_scenarios(path: Path) -> List[Scenario]:
    if path.is_file():
        return load_scenarios(path)
    if not path.exists():
        raise FileNotFoundError(path)
    scenarios: List[Scenario] = []
    for json_file in sorted(path.glob("*.json")):
        scenarios.extend(load_scenarios(json_file))
    if not scenarios:
        raise ValueError(f"No scenario JSON files discovered in {path}")
    return scenarios


def build_branchmind_namespace(args: argparse.Namespace, state_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        api_key=args.api_key,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version or DEFAULT_API_VERSION,
        cm_model=args.branchmind_cm_model or load_env("CM_MODEL") or DEFAULT_CM_MODEL,
        task_model=args.branchmind_task_model or load_env("TASK_MODEL") or DEFAULT_TASK_MODEL,
        summarizer_model=args.branchmind_summarizer_model
        or load_env("SUMMARIZER_MODEL")
        or DEFAULT_SUMMARIZER_MODEL,
        state_path=str(state_path),
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
    )


def create_branchmind_manager(args: argparse.Namespace, state_path: Path) -> DialogueManager:
    namespace = build_branchmind_namespace(args, state_path)
    config = AppConfig.from_args(namespace)
    if state_path.exists():
        state_path.unlink()
    return DialogueManager(config)


def create_baseline(args: argparse.Namespace, client) -> SlidingWindowBaseline:
    model_name = (
        args.baseline_model
        or load_env("BASELINE_MODEL")
        or load_env("TASK_MODEL")
        or DEFAULT_TASK_MODEL
    )
    summarizer_name = (
        args.baseline_summarizer_model
        or load_env("BASELINE_SUMMARIZER")
        or load_env("SUMMARIZER_MODEL")
        or DEFAULT_SUMMARIZER_MODEL
    )
    retry = LLMRetryConfig(args.retry_attempts, args.retry_delay)
    main_runner = AzureLLM(client, model_name, retry=retry)
    summarizer_runner = AzureLLM(client, summarizer_name, retry=retry)
    return SlidingWindowBaseline(
        llm=main_runner,
        summarizer=summarizer_runner,
        window_turns=args.baseline_window_turns,
        summary_trigger=args.baseline_summary_trigger,
        system_prompt=args.baseline_system_prompt,
    )


def scenario_already_processed(scenario_dir: Path) -> bool:
    branchmind_file = scenario_dir / "result_branchmind.json"
    baseline_file = scenario_dir / "result_baseline.json"
    return branchmind_file.exists() and baseline_file.exists()


def build_transcript_turn(
    role: str,
    content: str,
    turn: ScenarioTurn,
    *,
    extra_meta: Optional[Dict[str, str]] = None,
) -> TranscriptTurn:
    meta = {
        "turn_id": turn.turn_id,
        "thread_id": turn.thread_id,
    }
    if turn.timestamp_hint:
        meta["timestamp_hint"] = turn.timestamp_hint
    if turn.escalation_trigger:
        meta["escalation_trigger"] = turn.escalation_trigger
    if extra_meta:
        meta.update(extra_meta)
    if turn.expected_assistant_focus:
        meta["expected_assistant_focus"] = turn.expected_assistant_focus
    return TranscriptTurn(role=role, content=content, meta=meta)


def run_single_scenario(
    scenario: Scenario,
    args: argparse.Namespace,
    branchmind_state_root: Path,
    baseline_client,
) -> Dict[str, str]:
    scenario_dir = args.output_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    dump_json(scenario_dir / "scenario.json", scenario.to_dict())
    state_path = branchmind_state_root / f"{scenario.scenario_id}_{DEFAULT_STATE_FILE}"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    manager = create_branchmind_manager(args, state_path)
    baseline = create_baseline(args, baseline_client)

    branchmind_transcript = TranscriptBundle(
        assistant_label="branchmind",
        scenario_id=scenario.scenario_id,
        model_name=manager.config.task_model,
        system_description="BranchMind tree-structured dialogue manager",
        turns=[],
    )
    baseline_transcript = TranscriptBundle(
        assistant_label="baseline",
        scenario_id=scenario.scenario_id,
        model_name=baseline.llm.default_model,
        system_description="Sliding window baseline with periodic summarization",
        turns=[],
    )

    for turn in scenario.dialogue:
        branchmind_transcript.turns.append(build_transcript_turn("user", turn.user_message, turn))
        baseline_transcript.turns.append(build_transcript_turn("user", turn.user_message, turn))

        branchmind_response = manager.handle_request(turn.user_message, thread_id=turn.thread_id)
        baseline_response = baseline.handle_request(turn.user_message)

        branchmind_transcript.turns.append(
            TranscriptTurn(role="assistant", content=branchmind_response, meta={"source": "branchmind"})
        )
        baseline_transcript.turns.append(
            TranscriptTurn(role="assistant", content=baseline_response, meta={"source": "baseline"})
        )

    manager.save_state()

    branchmind_path = scenario_dir / "result_branchmind.json"
    baseline_path = scenario_dir / "result_baseline.json"
    dump_json(branchmind_path, branchmind_transcript.to_dict())
    dump_json(baseline_path, baseline_transcript.to_dict())

    summary_payload = {
        "scenario_id": scenario.scenario_id,
        "title": scenario.title,
        "branchmind_transcript": str(branchmind_path),
        "baseline_transcript": str(baseline_path),
        "branchmind_state": str(state_path),
    }
    dump_json(scenario_dir / "metadata.json", summary_payload)
    return summary_payload


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    GlobalConfig.env_overrides = {
        key: value
        for key, value in vars(args).items()
        if key in {"api_key", "azure_endpoint", "api_version"} and value
    }

    scenarios = collect_scenarios(args.scenarios)
    if args.scenario_limit:
        scenarios = scenarios[: args.scenario_limit]
    print(f"[runner] Loaded {len(scenarios)} scenarios.")

    branchmind_state_root = args.branchmind_state_root or (args.output_dir / "branchmind_states")
    branchmind_state_root.mkdir(parents=True, exist_ok=True)

    client = build_azure_client(args)

    manifest: List[Dict[str, str]] = []
    for scenario in scenarios:
        scenario_dir = args.output_dir / scenario.scenario_id
        if args.resume and scenario_already_processed(scenario_dir):
            metadata_path = scenario_dir / "metadata.json"
            if metadata_path.exists():
                manifest.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            print(f"[runner] Skipping {scenario.scenario_id} (resume).")
            continue
        print(f"[runner] Executing scenario {scenario.scenario_id} — {scenario.title}")
        metadata = run_single_scenario(scenario, args, branchmind_state_root, client)
        manifest.append(metadata)

    dump_json(args.output_dir / "results_index.json", manifest)
    print(f"[runner] Completed {len(manifest)} scenarios. Index written to results_index.json.")


if __name__ == "__main__":
    main()
