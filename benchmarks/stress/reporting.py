import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.common import dump_json  # type: ignore  # noqa: E402
from benchmarks.stress.run_stress_suite import (  # type: ignore  # noqa: E402
    ASSISTANT_BRANCHMIND,
    ASSISTANT_FULL_HISTORY,
    ASSISTANT_SLIDING,
    SUPPORTED_ASSISTANTS,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate stress-test metrics and render latency/token/tree-depth visualisations."
    )
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with stress suite outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store derived reports (defaults to <results-dir>/report).",
    )
    parser.add_argument(
        "--assistants",
        nargs="+",
        choices=sorted(SUPPORTED_ASSISTANTS),
        default=sorted(SUPPORTED_ASSISTANTS),
        help="Assistant labels to include.",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip matplotlib chart generation (useful if matplotlib is unavailable).",
    )
    return parser.parse_args()


def collect_scenario_dirs(results_dir: Path) -> List[Path]:
    scenario_dirs: List[Path] = []
    for path in sorted(results_dir.iterdir()):
        if path.is_dir() and (path / "scenario.json").exists():
            scenario_dirs.append(path)
    return scenario_dirs


def load_metrics(metrics_path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not metrics_path.exists():
        return records
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarise_metrics(records: List[Dict]) -> Dict[str, float]:
    if not records:
        return {"rounds": 0, "avg_latency": 0.0, "avg_tokens": 0.0, "total_tokens": 0.0}
    latency_sum = sum(item.get("latency", 0.0) for item in records)
    token_sum = sum(item.get("total_tokens", 0.0) for item in records)
    return {
        "rounds": len(records),
        "avg_latency": latency_sum / len(records),
        "avg_tokens": token_sum / len(records),
        "total_tokens": token_sum,
    }


def extract_branchmind_depth(records: List[Dict]) -> List[Optional[int]]:
    depths: List[Optional[int]] = []
    for item in records:
        snapshot = item.get("tree_snapshot") or {}
        depth = snapshot.get("max_depth")
        depths.append(depth if depth is not None else None)
    return depths


def build_chart(
    scenario_id: str,
    assistant_records: Dict[str, List[Dict]],
    output_dir: Path,
) -> None:
    if plt is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    round_count = 0

    for assistant, records in assistant_records.items():
        if not records:
            continue
        rounds = [item.get("round_index", idx + 1) for idx, item in enumerate(records)]
        round_count = max(round_count, len(rounds))
        latencies = [item.get("latency", math.nan) for item in records]
        tokens = [item.get("total_tokens", math.nan) for item in records]

        axes[0].plot(rounds, latencies, marker="o", label=assistant)
        axes[1].plot(rounds, tokens, marker="o", label=assistant)

        if assistant == ASSISTANT_BRANCHMIND:
            depths = extract_branchmind_depth(records)
            axes[2].plot(
                rounds,
                depths,
                marker="o",
                label="tree_depth",
            )

    axes[0].set_ylabel("Latency (s)")
    axes[0].set_title(f"{scenario_id}: Response Latency per Turn")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Total Tokens")
    axes[1].set_title("Token Usage per Turn")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel("Tree Depth (BranchMind)")
    axes[2].set_xlabel("Turn")
    axes[2].set_title("BranchMind Tree Growth")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    figure.savefig(output_dir / f"{scenario_id}_metrics.png", dpi=160)
    plt.close(figure)


def aggregate_assistant_metrics(
    scenarios: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    aggregates: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for entry in scenarios:
        for assistant, stats in entry.items():
            if assistant not in aggregates:
                aggregates[assistant] = {"avg_latency": 0.0, "avg_tokens": 0.0, "total_tokens": 0.0, "samples": 0}
            aggregates[assistant]["avg_latency"] += stats.get("avg_latency", 0.0)
            aggregates[assistant]["avg_tokens"] += stats.get("avg_tokens", 0.0)
            aggregates[assistant]["total_tokens"] += stats.get("total_tokens", 0.0)
            counts[assistant] = counts.get(assistant, 0) + 1

    for assistant, stats in aggregates.items():
        sample_count = counts.get(assistant, 1)
        stats["avg_latency"] = stats["avg_latency"] / sample_count
        stats["avg_tokens"] = stats["avg_tokens"] / sample_count
        stats["samples"] = sample_count
    return aggregates


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.results_dir / "report")
    output_dir.mkdir(parents=True, exist_ok=True)

    if plt is None and not args.no_charts:
        print("⚠️ matplotlib not available; charts will be skipped. Install matplotlib to enable visualisations.")

    scenario_dirs = collect_scenario_dirs(args.results_dir)
    per_scenario_summary: List[Dict[str, Dict[str, float]]] = []

    for scenario_dir in scenario_dirs:
        scenario_id = scenario_dir.name
        assistant_records: Dict[str, List[Dict]] = {}
        scenario_summary: Dict[str, Dict[str, float]] = {}

        for assistant in args.assistants:
            metrics_path = scenario_dir / assistant / "metrics.jsonl"
            records = load_metrics(metrics_path)
            if not records:
                continue
            assistant_records[assistant] = records
            scenario_summary[assistant] = summarise_metrics(records)

        if not scenario_summary:
            continue

        per_scenario_summary.append(scenario_summary)

        scenario_output = output_dir / scenario_id
        scenario_output.mkdir(parents=True, exist_ok=True)
        dump_json(scenario_output / "summary.json", scenario_summary)

        if not args.no_charts and plt is not None:
            build_chart(scenario_id, assistant_records, scenario_output)

    aggregate = aggregate_assistant_metrics(per_scenario_summary)
    dump_json(output_dir / "aggregate_summary.json", {"assistants": aggregate, "scenarios": len(per_scenario_summary)})

    print(f"[report] Generated summaries for {len(per_scenario_summary)} scenarios. Output: {output_dir}")


if __name__ == "__main__":
    main()
