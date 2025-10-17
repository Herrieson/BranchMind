import argparse
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.common import (
    AzureLLM,
    GlobalConfig,
    LLMRetryConfig,
    Scenario,
    build_azure_client,
    dump_json,
    load_env,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multi-threaded scenarios via an LLM.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the combined scenarios JSON file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=6,
        help="Number of scenarios to request from the generator LLM (5–10 recommended).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Azure OpenAI model used for generation (defaults to GENERATOR_MODEL or TASK_MODEL).",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("prompts/generator_prompt.md"),
        help="Path to the generator prompt template.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=None,
        help="Optional directory to emit one JSON file per scenario.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        default=3,
        help="LLM retry attempts.",
    )
    parser.add_argument("--retry-delay", type=float, default=2.0, help="Delay between LLM retries in seconds.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Azure OpenAI API key override.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Azure OpenAI endpoint override.",
    )
    parser.add_argument(
        "--api-version",
        default=None,
        help="Azure OpenAI API version override.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting numeric suffix for generated scenario IDs (e.g., 101 -> S-101).",
    )
    return parser.parse_args()


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_count(scenarios: List[Scenario], expected: int) -> None:
    if len(scenarios) != expected:
        raise ValueError(f"Expected {expected} scenarios, got {len(scenarios)}.")


def build_single_prompt(base_prompt: str, scenario_id: str, index: int, total: int) -> str:
    guidance = (
        "\n\nIMPORTANT: Ignore any prior instruction about generating multiple scenarios at once. "
        "For this call return JSON with a `scenarios` array containing exactly one scenario object. "
        f"Use `scenario_id` \"{scenario_id}\" and remember this is scenario {index} of {total}. "
        "Do not include commentary or extra keys."
    )
    return base_prompt.rstrip() + guidance


def parse_single_scenario(payload: dict, expected_id: str) -> Scenario:
    if "scenarios" in payload:
        items = payload["scenarios"]
        if not isinstance(items, list):
            raise ValueError("LLM response has 'scenarios' but it is not a list.")
        if len(items) != 1:
            raise ValueError(f"Expected exactly one scenario in response, received {len(items)}.")
        scenario_payload = items[0]
    elif "scenario" in payload:
        scenario_payload = payload["scenario"]
    else:
        scenario_payload = payload

    if not isinstance(scenario_payload, dict):
        raise ValueError("Scenario payload must be a JSON object.")

    scenario_payload["scenario_id"] = expected_id
    return Scenario.from_dict(scenario_payload)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    GlobalConfig.env_overrides = {k: v for k, v in vars(args).items() if k in {"api_key", "azure_endpoint", "api_version"} and v}

    base_prompt = load_prompt(args.prompt_path)
    client = build_azure_client(args)

    model_name = args.model or load_env("GENERATOR_MODEL") or load_env("TASK_MODEL") or "gpt-4o"
    runner = AzureLLM(client, default_model=model_name, retry=LLMRetryConfig(args.retry_attempts, args.retry_delay))

    scenarios: List[Scenario] = []
    for index in range(args.count):
        scenario_id = f"S-{args.start_index + index:03d}"
        prompt_text = build_single_prompt(base_prompt, scenario_id, index + 1, args.count)
        messages = [
            {
                "role": "system",
                "content": "You are a rigorous scenario generator for BranchMind evaluations. Follow the instructions exactly.",
            },
            {"role": "user", "content": prompt_text},
        ]

        print(f"[generator] Requesting scenario {index + 1}/{args.count} ({scenario_id}) with model '{model_name}'.")

        last_error: Optional[Tuple[int, Exception]] = None
        for attempt in range(1, args.retry_attempts + 1):
            try:
                response = runner.chat(messages, json_mode=True)
                payload = json.loads(response)
                scenario = parse_single_scenario(payload, scenario_id)
                scenarios.append(scenario)
                break
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = (attempt, exc)
                print(f"[generator] Attempt {attempt}/{args.retry_attempts} failed for {scenario_id}: {exc}")
                if attempt == args.retry_attempts:
                    raise ValueError(f"Failed to generate scenario {scenario_id}") from exc
                time.sleep(args.retry_delay)

        if last_error and last_error[0] < args.retry_attempts:
            print(f"[generator] Retried scenario {scenario_id} after validation failure.")

    ensure_count(scenarios, args.count)

    dump_json(args.output, {"scenarios": [scenario.to_dict() for scenario in scenarios]})
    print(f"[generator] Wrote combined scenarios to {args.output} (count={len(scenarios)}).")

    if args.split_dir:
        for scenario in scenarios:
            scenario_path = args.split_dir / f"{scenario.scenario_id}.json"
            dump_json(scenario_path, scenario.to_dict())
            print(f"[generator] Wrote {scenario_path}.")


if __name__ == "__main__":
    main()
