import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.common import (
    AzureLLM,
    GlobalConfig,
    LLMRetryConfig,
    build_azure_client,
    choose_random_label,
    dump_json,
    load_env,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind evaluation of BranchMind vs baseline transcripts.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing scenario subdirectories produced by run_evaluation.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store evaluator outputs (defaults to <results-dir>/judgement).",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompts/evaluator_prompt.md"),
        help="Prompt template for the evaluator LLM.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override evaluator model (defaults to EVALUATOR_MODEL or TASK_MODEL).",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="Evaluator retry attempts.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Delay between evaluator retries (seconds).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios that already have a judgement file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic assistant shuffling.",
    )
    parser.add_argument("--api-key", default=None, help="Azure OpenAI API key override.")
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint override.")
    parser.add_argument("--api-version", default=None, help="Azure OpenAI API version override.")
    return parser.parse_args()


def collect_scenario_dirs(results_dir: Path) -> List[Path]:
    """Return only scenario directories that contain the required payload files."""
    scenario_dirs: List[Path] = []
    for path in sorted(results_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "scenario.json").exists():
            scenario_dirs.append(path)
    return scenario_dirs


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def randomize_assignments(
    branchmind_payload: Dict,
    baseline_payload: Dict,
) -> Tuple[Dict, Dict, Dict]:
    if choose_random_label():
        return (
            branchmind_payload,
            baseline_payload,
            {"assistant_a": "branchmind", "assistant_b": "baseline"},
        )
    return (
        baseline_payload,
        branchmind_payload,
        {"assistant_a": "baseline", "assistant_b": "branchmind"},
    )


def render_prompt(template: str, scenario: Dict, assistant_a: Dict, assistant_b: Dict) -> str:
    return (
        f"{template.rstrip()}\n\n"
        "SCENARIO_JSON:\n"
        f"{json.dumps(scenario, ensure_ascii=False, indent=2)}\n\n"
        "ASSISTANT_A_JSON:\n"
        f"{json.dumps(assistant_a, ensure_ascii=False, indent=2)}\n\n"
        "ASSISTANT_B_JSON:\n"
        f"{json.dumps(assistant_b, ensure_ascii=False, indent=2)}"
    )


def compute_aggregate(stats: List[Dict]) -> Dict:
    if not stats:
        return {}
    branchmind_scores = [item["weighted_scores"]["branchmind"] for item in stats]
    baseline_scores = [item["weighted_scores"]["baseline"] for item in stats]
    wins = {"branchmind": 0, "baseline": 0, "tie": 0}
    for item in stats:
        wins[item["winner_actual"]] += 1
    total = len(stats)
    preference_rate = wins["branchmind"] / total if total else 0.0

    return {
        "samples": total,
        "branchmind_average": sum(branchmind_scores) / total if total else 0.0,
        "baseline_average": sum(baseline_scores) / total if total else 0.0,
        "branchmind_win_rate": preference_rate,
        "win_counts": wins,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    GlobalConfig.env_overrides = {
        key: value
        for key, value in vars(args).items()
        if key in {"api_key", "azure_endpoint", "api_version"} and value
    }

    output_dir = args.output_dir or (args.results_dir / "judgement")
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_text = args.prompt_path.read_text(encoding="utf-8")
    client = build_azure_client(args)
    model_name = args.model or load_env("EVALUATOR_MODEL") or load_env("TASK_MODEL") or "gpt-4o"
    evaluator = AzureLLM(client, model_name, retry=LLMRetryConfig(args.retry_attempts, args.retry_delay))

    scenario_dirs = collect_scenario_dirs(args.results_dir)
    print(f"[judge] Found {len(scenario_dirs)} scenario directories.")

    aggregate_entries: List[Dict] = []

    for scenario_dir in scenario_dirs:
        scenario_id = scenario_dir.name
        output_path = output_dir / f"{scenario_id}_judgement.json"
        if args.resume and output_path.exists():
            print(f"[judge] Skipping {scenario_id} (resume).")
            aggregate_entries.append(load_json(output_path))
            continue

        scenario = load_json(scenario_dir / "scenario.json")
        branchmind_transcript = load_json(scenario_dir / "result_branchmind.json")
        baseline_transcript = load_json(scenario_dir / "result_baseline.json")

        assistant_a, assistant_b, mapping = randomize_assignments(branchmind_transcript, baseline_transcript)
        rendered_prompt = render_prompt(prompt_text, scenario, assistant_a, assistant_b)

        messages = [
            {"role": "system", "content": "You are an impartial evaluator following instructions exactly."},
            {"role": "user", "content": rendered_prompt},
        ]
        print(f"[judge] Evaluating scenario {scenario_id} with model '{model_name}'.")
        response_text = evaluator.chat(messages, json_mode=True)
        judgement = json.loads(response_text)

        weighted_scores = {
            "assistant_a": judgement.get("weighted_averages", {}).get("assistant_a", 0.0),
            "assistant_b": judgement.get("weighted_averages", {}).get("assistant_b", 0.0),
        }

        winner_label = judgement.get("winner", "tie")
        actual_winner = mapping.get(winner_label, "tie")
        branchmind_score = (
            weighted_scores["assistant_a"]
            if mapping["assistant_a"] == "branchmind"
            else weighted_scores["assistant_b"]
        )
        baseline_score = (
            weighted_scores["assistant_a"]
            if mapping["assistant_a"] == "baseline"
            else weighted_scores["assistant_b"]
        )

        record = {
            "scenario_id": scenario_id,
            "scenario_title": scenario.get("title"),
            "winner_reported": winner_label,
            "winner_actual": actual_winner,
            "mapping": mapping,
            "weighted_scores": {
                "branchmind": branchmind_score,
                "baseline": baseline_score,
            },
            "judgement": judgement,
        }
        dump_json(output_path, record)
        aggregate_entries.append(record)

    summary = compute_aggregate(aggregate_entries)
    dump_json(output_dir / "summary.json", {"summary": summary, "scenarios": aggregate_entries})
    print(f"[judge] Completed evaluations. Summary written to {output_dir / 'summary.json'}.")


if __name__ == "__main__":
    main()
