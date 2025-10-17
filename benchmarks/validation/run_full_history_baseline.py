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
    TranscriptBundle,
    TranscriptTurn,
    build_azure_client,
    dump_json,
    load_env,
    seed_everything,
)
from benchmarks.validation.run_evaluation import (
    build_transcript_turn,
    collect_scenarios,
    create_branchmind_manager,
)
from branchmind import DEFAULT_RETRY_ATTEMPTS, DEFAULT_RETRY_DELAY, DEFAULT_STATE_FILE


class FullHistoryBaseline:
    """Baseline that replays the full conversation history without extra summarization."""

    def __init__(self, llm: AzureLLM, *, system_prompt: str) -> None:
        self.llm = llm
        self.system_prompt = system_prompt.strip() or "You are a helpful assistant."
        self.dialogue: List[Dict[str, str]] = []

    def handle_request(
        self,
        query: str,
        *,
        telemetry: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> str:
        def emit(label: str, payload: Dict[str, Any]) -> None:
            if telemetry:
                telemetry(label, payload)

        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.dialogue)
        messages.append({"role": "user", "content": query})
        response = self.llm.chat(messages, telemetry=lambda data: emit("main", data))
        self.dialogue.append({"role": "user", "content": query})
        self.dialogue.append({"role": "assistant", "content": response})
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute BranchMind vs full-history baseline (no summarization)."
    )
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
        "--baseline-system-prompt",
        default="You are a diligent assistant with access to the full dialogue history.",
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
    parser.add_argument(
        "--results-index-name",
        default="results_index_full_history.json",
        help="Filename for the manifest summarising generated artifacts.",
    )
    return parser.parse_args()


def create_full_history_baseline(args: argparse.Namespace, client) -> FullHistoryBaseline:
    model_name = (
        args.baseline_model
        or load_env("FULL_HISTORY_BASELINE_MODEL")
        or load_env("BASELINE_MODEL")
        or load_env("TASK_MODEL")
    )
    if not model_name:
        raise EnvironmentError(
            "Baseline model not provided. Set --baseline-model or define FULL_HISTORY_BASELINE_MODEL / "
            "BASELINE_MODEL / TASK_MODEL."
        )
    retry = LLMRetryConfig(args.retry_attempts, args.retry_delay)
    main_runner = AzureLLM(client, model_name, retry=retry)
    return FullHistoryBaseline(llm=main_runner, system_prompt=args.baseline_system_prompt)


def scenario_already_processed(
    scenario_dir: Path, *, baseline_filename: str
) -> bool:
    branchmind_file = scenario_dir / "result_branchmind.json"
    baseline_file = scenario_dir / baseline_filename
    return branchmind_file.exists() and baseline_file.exists()


def merge_metadata(
    scenario_dir: Path,
    *,
    baseline_key: str,
    baseline_path: Path,
) -> Dict[str, str]:
    metadata_path = scenario_dir / "metadata.json"
    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        payload = {}
    payload[baseline_key] = str(baseline_path)
    dump_json(metadata_path, payload)
    return payload


def run_single_scenario(
    scenario: Scenario,
    args: argparse.Namespace,
    branchmind_state_root: Path,
    baseline_client,
    *,
    baseline_filename: str,
    baseline_label: str,
    baseline_description: str,
    baseline_key: str,
) -> Dict[str, str]:
    scenario_dir = args.output_dir / scenario.scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    dump_json(scenario_dir / "scenario.json", scenario.to_dict())
    state_path = branchmind_state_root / f"{scenario.scenario_id}_{DEFAULT_STATE_FILE}"
    state_path.parent.mkdir(parents=True, exist_ok=True)

    manager = create_branchmind_manager(args, state_path)
    baseline = create_full_history_baseline(args, baseline_client)

    branchmind_transcript = TranscriptBundle(
        assistant_label="branchmind",
        scenario_id=scenario.scenario_id,
        model_name=manager.config.task_model,
        system_description="BranchMind tree-structured dialogue manager",
        turns=[],
    )
    baseline_transcript = TranscriptBundle(
        assistant_label=baseline_label,
        scenario_id=scenario.scenario_id,
        model_name=baseline.llm.default_model,
        system_description=baseline_description,
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
            TranscriptTurn(role="assistant", content=baseline_response, meta={"source": baseline_label})
        )

    manager.save_state()

    branchmind_path = scenario_dir / "result_branchmind.json"
    baseline_path = scenario_dir / baseline_filename

    dump_json(branchmind_path, branchmind_transcript.to_dict())
    dump_json(baseline_path, baseline_transcript.to_dict())

    payload = merge_metadata(
        scenario_dir,
        baseline_key=baseline_key,
        baseline_path=baseline_path,
    )
    payload.update(
        {
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "branchmind_transcript": str(branchmind_path),
        }
    )
    payload["branchmind_state"] = str(state_path)
    return payload


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
    print(f"[full-history] Loaded {len(scenarios)} scenarios.")

    branchmind_state_root = args.branchmind_state_root or (args.output_dir / "branchmind_states")
    branchmind_state_root.mkdir(parents=True, exist_ok=True)

    client = build_azure_client(args)
    baseline_filename = "result_baseline_full_history.json"
    baseline_label = "baseline_full_history"
    baseline_description = "Full-history baseline (no summarization or windowing)"
    baseline_key = "baseline_full_history_transcript"

    manifest: List[Dict[str, str]] = []
    for scenario in scenarios:
        scenario_dir = args.output_dir / scenario.scenario_id
        if args.resume and scenario_already_processed(scenario_dir, baseline_filename=baseline_filename):
            metadata_path = scenario_dir / "metadata.json"
            if metadata_path.exists():
                manifest.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            print(f"[full-history] Skipping {scenario.scenario_id} (resume).")
            continue
        print(f"[full-history] Executing scenario {scenario.scenario_id} — {scenario.title}")
        metadata = run_single_scenario(
            scenario,
            args,
            branchmind_state_root,
            client,
            baseline_filename=baseline_filename,
            baseline_label=baseline_label,
            baseline_description=baseline_description,
            baseline_key=baseline_key,
        )
        manifest.append(metadata)

    index_path = args.output_dir / args.results_index_name
    dump_json(index_path, manifest)
    print(f"[full-history] Completed {len(manifest)} scenarios. Index written to {index_path.name}.")


if __name__ == "__main__":
    main()
