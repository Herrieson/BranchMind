import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from benchmarks.common import (
    AzureLLM,
    GlobalConfig,
    LLMRetryConfig,
    Scenario,
    ScenarioNotes,
    ScenarioRubricItem,
    ScenarioThread,
    ScenarioTurn,
    build_azure_client,
    dump_json,
    load_env,
    seed_everything,
)


PLAN_PROMPT_PATH = Path("benchmarks/stress/prompts/stress_plan_prompt.md")
CHUNK_PROMPT_PATH = Path("benchmarks/stress/prompts/stress_chunk_prompt.md")
METADATA_PROMPT_PATH = Path("benchmarks/stress/prompts/stress_metadata_prompt.md")


@dataclass
class ChunkPlan:
    chunk_id: str
    start_turn_index: int
    turn_count: int
    focus_threads: List[str]
    user_goals: List[str]
    escalations: List[str]
    notes: str
    allow_early_termination: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Iteratively expand a stress outline into a Scenario JSON.")
    parser.add_argument("outline", type=Path, help="Path to Markdown/JSON outline file.")
    parser.add_argument("output", type=Path, help="Where to write the generated scenario JSON.")
    parser.add_argument("--scenario-id", default=None, help="Override scenario_id (defaults to plan suggestion or output stem).")
    parser.add_argument("--target-turns", type=int, default=90, help="Desired total user turns (50–100 recommended).")
    parser.add_argument("--chunk-size", type=int, default=12, help="Approximate number of turns per generation chunk.")
    parser.add_argument("--context-turns", type=int, default=12, help="Transcript turns to feed back into each chunk prompt.")

    parser.add_argument("--plan-prompt", type=Path, default=PLAN_PROMPT_PATH)
    parser.add_argument("--chunk-prompt", type=Path, default=CHUNK_PROMPT_PATH)
    parser.add_argument("--metadata-prompt", type=Path, default=METADATA_PROMPT_PATH)

    parser.add_argument("--plan-model", default=None, help="Model for planning phase (defaults to STRESS_PLAN_MODEL/GENERATOR_MODEL/TASK_MODEL).")
    parser.add_argument("--chunk-model", default=None, help="Model for chunk generation (defaults to STRESS_CHUNK_MODEL/TASK_MODEL).")
    parser.add_argument(
        "--metadata-model",
        default=None,
        help="Model for rubric/notes generation (defaults to STRESS_METADATA_MODEL/SUMMARIZER_MODEL/TASK_MODEL).",
    )

    parser.add_argument("--retry-attempts", type=int, default=LLMRetryConfig().attempts, help="LLM retry attempts per request.")
    parser.add_argument("--retry-delay", type=float, default=LLMRetryConfig().delay, help="Delay between retries (seconds).")
    parser.add_argument("--parse-retries", type=int, default=3, help="Retries when JSON parsing/validation fails.")

    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing plan/output when present.")
    parser.add_argument("--dry-run", action="store_true", help="Generate the plan only, do not write scenario.")
    parser.add_argument("--skip-metadata", action="store_true", help="Skip rubric/notes synthesis (useful for drafts).")
    parser.add_argument("--verbose", action="store_true", help="Print verbose progress information.")

    parser.add_argument("--api-key", default=None, help="Azure OpenAI API key override.")
    parser.add_argument("--azure-endpoint", default=None, help="Azure OpenAI endpoint override.")
    parser.add_argument("--api-version", default=None, help="Azure OpenAI API version override.")
    return parser.parse_args()


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ScenarioBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.outline_path = args.outline
        self.output_path = args.output
        self.plan_path = self.output_path.with_suffix(".plan.json")
        self.retry_delay = max(0.0, args.retry_delay)
        self.parse_retries = max(1, args.parse_retries)
        self.context_turns = max(0, args.context_turns)
        self.target_turns = max(10, args.target_turns)
        self.chunk_size = max(4, args.chunk_size)
        self.verbose = args.verbose

        seed_everything(args.seed)
        GlobalConfig.env_overrides = {
            key: value
            for key, value in {
                "API_KEY": args.api_key,
                "AZURE_ENDPOINT": args.azure_endpoint,
                "API_VERSION": args.api_version,
            }.items()
            if value
        }

        plan_model = (
            args.plan_model
            or load_env("STRESS_PLAN_MODEL")
            or load_env("GENERATOR_MODEL")
            or load_env("TASK_MODEL")
            or "gpt-4o"
        )
        chunk_model = (
            args.chunk_model
            or load_env("STRESS_CHUNK_MODEL")
            or load_env("TASK_MODEL")
            or "gpt-4o"
        )
        metadata_model = (
            args.metadata_model
            or load_env("STRESS_METADATA_MODEL")
            or load_env("SUMMARIZER_MODEL")
            or load_env("TASK_MODEL")
            or "gpt-4o-mini"
        )

        client = build_azure_client(args)
        retry_cfg = LLMRetryConfig(args.retry_attempts, args.retry_delay)
        self.plan_runner = AzureLLM(client, default_model=plan_model, retry=retry_cfg)
        self.chunk_runner = AzureLLM(client, default_model=chunk_model, retry=retry_cfg)
        self.metadata_runner = AzureLLM(client, default_model=metadata_model, retry=retry_cfg)

        self.plan_prompt_template = load_text(args.plan_prompt)
        self.chunk_prompt_template = load_text(args.chunk_prompt)
        self.metadata_prompt_template = load_text(args.metadata_prompt)

        self.outline_text = load_text(self.outline_path)
        self.plan_payload: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    def run(self) -> None:
        scenario = self._load_existing_scenario() if self.args.resume else None

        self.plan_payload = self._load_existing_plan() if self.args.resume else None
        if self.plan_payload is None:
            self.plan_payload = self._generate_plan()
            self._write_plan(self.plan_payload)

        if self.args.dry_run:
            print(json.dumps(self.plan_payload, ensure_ascii=False, indent=2))
            return

        if scenario is None:
            scenario = self._initialise_scenario(self.plan_payload)
        else:
            if self.verbose:
                print(f"[resume] Loaded existing scenario with {len(scenario.dialogue)} turns.")

        chunk_plans = self._extract_chunks(self.plan_payload)
        generated_turns = len(scenario.dialogue)

        for chunk in chunk_plans:
            expected_start = chunk.start_turn_index
            expected_end = chunk.start_turn_index + chunk.turn_count - 1
            if generated_turns >= expected_end:
                if self.verbose:
                    print(f"[skip] Chunk {chunk.chunk_id} already satisfied ({generated_turns} turns).")
                continue
            if generated_turns + 1 > expected_start:
                raise RuntimeError(
                    f"Cannot resume mid-chunk ({chunk.chunk_id}). Existing turns={generated_turns}, "
                    f"chunk expected start={expected_start}."
                )
            scenario.dialogue.extend(
                self._generate_chunk(chunk, scenario, len(scenario.dialogue))
            )
            generated_turns = len(scenario.dialogue)
            self._write_partial(scenario)

        if not self.args.skip_metadata:
            evaluation, notes = self._generate_metadata(scenario)
            scenario.evaluation_rubric = evaluation
            scenario.notes = notes
        else:
            if self.verbose:
                print("[info] Metadata synthesis skipped by flag.")

        self._write_final(scenario)

    # ------------------------------------------------------------------
    def _load_existing_plan(self) -> Optional[Dict[str, Any]]:
        if not self.plan_path.exists():
            return None
        try:
            payload = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if self.verbose:
                print(f"[resume] Loaded existing plan from {self.plan_path}.")
            return payload
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse existing plan at {self.plan_path}: {exc}") from exc

    def _load_existing_scenario(self) -> Optional[Scenario]:
        if not self.output_path.exists():
            return None
        try:
            payload = json.loads(self.output_path.read_text(encoding="utf-8"))
            scenarios = payload["scenarios"] if isinstance(payload, dict) and "scenarios" in payload else payload
            if isinstance(scenarios, list) and scenarios:
                scenario = Scenario.from_dict(scenarios[0])
            else:
                scenario = Scenario.from_dict(payload)
            if self.verbose:
                print(f"[resume] Loaded scenario from {self.output_path}.")
            return scenario
        except Exception as exc:
            raise RuntimeError(f"Failed to parse existing scenario at {self.output_path}: {exc}") from exc

    def _generate_plan(self) -> Dict[str, Any]:
        if self.verbose:
            print("[plan] Generating scenario plan...")
        prompt = (
            self.plan_prompt_template.replace("<<OUTLINE>>", self.outline_text)
            .replace("<<TARGET_TURNS>>", str(self.target_turns))
            .replace("<<CHUNK_SIZE>>", str(self.chunk_size))
        )
        messages = [
            {"role": "system", "content": "You are an expert scenario planner for BranchMind. Only output JSON."},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(1, self.parse_retries + 1):
            response = self.plan_runner.chat(messages, json_mode=True)
            try:
                payload = json.loads(response)
                return self._validate_and_normalise_plan(payload)
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == self.parse_retries:
                    raise RuntimeError(f"Failed to obtain valid plan: {exc}") from exc
                if self.verbose:
                    print(f"[plan] Validation failed (attempt {attempt}/{self.parse_retries}): {exc}")
                time.sleep(self.retry_delay)
        raise RuntimeError("Unreachable: plan generation loop exhausted without raising.")

    def _initialise_scenario(self, plan: Dict[str, Any]) -> Scenario:
        scenario_block = plan["scenario"]
        threads = [ScenarioThread.from_dict(item) for item in scenario_block["threads"]]
        scenario_id = (
            self.args.scenario_id
            or scenario_block.get("scenario_id")
            or self.output_path.stem.upper()
        )
        scenario = Scenario(
            scenario_id=scenario_id,
            title=scenario_block["title"],
            background=scenario_block["background"],
            user_profile=scenario_block["user_profile"],
            assistant_profile=scenario_block["assistant_profile"],
            threads=threads,
            dialogue=[],
            evaluation_rubric=[],
            notes=ScenarioNotes(),
        )
        if self.verbose:
            print(f"[plan] Scenario initialised with {len(threads)} threads (ID={scenario_id}).")
        return scenario

    def _extract_chunks(self, plan: Dict[str, Any]) -> List[ChunkPlan]:
        generation = plan["generation_plan"]
        chunks: List[ChunkPlan] = []
        for raw in generation["chunks"]:
            chunk = ChunkPlan(
                chunk_id=str(raw["chunk_id"]).strip(),
                start_turn_index=int(raw["start_turn_index"]),
                turn_count=int(raw["turn_count"]),
                focus_threads=[str(t).strip() for t in raw.get("focus_threads", []) if str(t).strip()],
                user_goals=[str(g).strip() for g in raw.get("user_goals", []) if str(g).strip()],
                escalations=[str(e).strip() for e in raw.get("escalations", []) if str(e).strip()],
                notes=str(raw.get("notes", "")).strip(),
                allow_early_termination=bool(raw.get("allow_early_termination", False)),
            )
            chunks.append(chunk)
        return chunks

    def _generate_chunk(
        self,
        chunk: ChunkPlan,
        scenario: Scenario,
        existing_turns: int,
    ) -> List[ScenarioTurn]:
        if self.verbose:
            print(f"[chunk] Generating {chunk.chunk_id} ({chunk.turn_count} turns).")
        prompt = self._render_chunk_prompt(chunk, scenario)
        messages = [
            {"role": "system", "content": "You author realistic user turns for BranchMind stress tests. Only output JSON."},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(1, self.parse_retries + 1):
            response = self.chunk_runner.chat(messages, json_mode=True)
            try:
                payload = json.loads(response)
                turns = self._parse_chunk_turns(payload, chunk, scenario)
                for turn_idx, turn in enumerate(turns, start=existing_turns + 1):
                    turn.turn_id = f"U{turn_idx}"
                if self.verbose:
                    summary = payload.get("summary", "")
                    print(f"[chunk] {chunk.chunk_id} produced {len(turns)} turns. {summary}")
                return turns
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == self.parse_retries:
                    raise RuntimeError(f"Failed to obtain valid chunk for {chunk.chunk_id}: {exc}") from exc
                if self.verbose:
                    print(
                        f"[chunk] Validation failed for {chunk.chunk_id} "
                        f"(attempt {attempt}/{self.parse_retries}): {exc}"
                    )
                time.sleep(self.retry_delay)
        raise RuntimeError("Unreachable: chunk generation loop exhausted without raising.")

    def _generate_metadata(self, scenario: Scenario) -> (List[ScenarioRubricItem], ScenarioNotes):
        if self.verbose:
            print("[meta] Synthesising evaluation rubric and notes...")
        prompt = self._render_metadata_prompt(scenario)
        messages = [
            {"role": "system", "content": "You derive evaluation rubrics for BranchMind stress tests. Only output JSON."},
            {"role": "user", "content": prompt},
        ]
        for attempt in range(1, self.parse_retries + 1):
            response = self.metadata_runner.chat(messages, json_mode=True)
            try:
                payload = json.loads(response)
                evaluation = [
                    ScenarioRubricItem.from_dict(item)
                    for item in payload.get("evaluation_rubric", [])
                ]
                if not evaluation:
                    raise ValueError("No evaluation rubric items returned.")
                total_weight = sum(item.weight for item in evaluation)
                if total_weight <= 0:
                    raise ValueError("Evaluation rubric weights must sum to > 0.")
                normalised = [
                    ScenarioRubricItem(
                        criterion=item.criterion,
                        description=item.description,
                        weight=round(item.weight / total_weight, 4),
                    )
                    for item in evaluation
                ]
                notes = ScenarioNotes.from_dict(payload.get("notes", {}))
                return normalised, notes
            except (json.JSONDecodeError, ValueError) as exc:
                if attempt == self.parse_retries:
                    raise RuntimeError(f"Failed to obtain metadata: {exc}") from exc
                if self.verbose:
                    print(f"[meta] Validation failed (attempt {attempt}/{self.parse_retries}): {exc}")
                time.sleep(self.retry_delay)
        raise RuntimeError("Unreachable: metadata generation loop exhausted without raising.")

    # ------------------------------------------------------------------
    def _write_plan(self, plan: Dict[str, Any]) -> None:
        dump_json(self.plan_path, plan)
        if self.verbose:
            print(f"[plan] Saved plan to {self.plan_path}.")

    def _write_partial(self, scenario: Scenario) -> None:
        payload = {"scenarios": [scenario.to_dict()]}
        dump_json(self.output_path, payload)
        if self.verbose:
            print(f"[progress] Wrote partial scenario with {len(scenario.dialogue)} turns to {self.output_path}.")

    def _write_final(self, scenario: Scenario) -> None:
        payload = {"scenarios": [scenario.to_dict()]}
        dump_json(self.output_path, payload)
        if self.verbose:
            print(f"[done] Scenario complete with {len(scenario.dialogue)} turns. Output -> {self.output_path}")

    # ------------------------------------------------------------------
    def _validate_and_normalise_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if "scenario" not in payload or "generation_plan" not in payload:
            raise ValueError("Plan response missing 'scenario' or 'generation_plan' keys.")
        scenario_block = payload["scenario"]
        required_fields = ["title", "background", "user_profile", "assistant_profile", "threads"]
        for field in required_fields:
            if not scenario_block.get(field):
                raise ValueError(f"Scenario plan missing required field '{field}'.")

        threads_raw = scenario_block["threads"]
        if not isinstance(threads_raw, list) or not threads_raw:
            raise ValueError("Scenario plan must include a non-empty 'threads' list.")
        seen_threads = set()
        normalised_threads = []
        for item in threads_raw:
            thread_id = str(item.get("thread_id", "")).strip()
            if not thread_id:
                raise ValueError("Thread missing 'thread_id'.")
            if thread_id in seen_threads:
                raise ValueError(f"Duplicate thread_id '{thread_id}' detected.")
            seen_threads.add(thread_id)
            intent = str(item.get("intent", "")).strip()
            if not intent:
                raise ValueError(f"Thread '{thread_id}' missing 'intent'.")
            priority = str(item.get("priority", "medium")).strip() or "medium"
            key_entities = [
                str(entity).strip() for entity in item.get("key_entities", []) if str(entity).strip()
            ]
            normalised_threads.append(
                {
                    "thread_id": thread_id,
                    "intent": intent,
                    "priority": priority,
                    "key_entities": key_entities,
                }
            )
        scenario_block["threads"] = normalised_threads

        generation = payload["generation_plan"]
        if not isinstance(generation, dict):
            raise ValueError("'generation_plan' must be an object.")
        target_turns = int(generation.get("target_turns", self.target_turns))
        if target_turns < 30 or target_turns > 120:
            raise ValueError(f"target_turns {target_turns} out of expected range (30-120).")
        generation["target_turns"] = target_turns
        generation["chunk_size"] = int(generation.get("chunk_size", self.chunk_size))
        chunks_raw = generation.get("chunks")
        if not isinstance(chunks_raw, list) or not chunks_raw:
            raise ValueError("generation_plan.chunks must be a non-empty list.")
        normalised_chunks = []
        cursor = 1
        accumulated = 0
        for raw in chunks_raw:
            chunk_id = str(raw.get("chunk_id", "")).strip()
            turn_count = int(raw.get("turn_count", generation["chunk_size"]))
            if turn_count <= 0:
                raise ValueError(f"Chunk '{chunk_id}' has non-positive turn_count.")
            start_index = int(raw.get("start_turn_index", cursor))
            focus_threads = [str(t).strip() for t in raw.get("focus_threads", []) if str(t).strip()]
            for thread_id in focus_threads:
                if thread_id not in seen_threads:
                    raise ValueError(f"Chunk '{chunk_id}' references unknown thread '{thread_id}'.")
            user_goals = [str(goal).strip() for goal in raw.get("user_goals", []) if str(goal).strip()]
            escalations = [str(e).strip() for e in raw.get("escalations", []) if str(e).strip()]
            notes = str(raw.get("notes", "")).strip()
            allow_early = bool(raw.get("allow_early_termination", False))
            normalised_chunks.append(
                {
                    "chunk_id": chunk_id or f"C{len(normalised_chunks)+1}",
                    "start_turn_index": start_index,
                    "turn_count": turn_count,
                    "focus_threads": focus_threads,
                    "user_goals": user_goals,
                    "escalations": escalations,
                    "notes": notes,
                    "allow_early_termination": allow_early,
                }
            )
            cursor = start_index + turn_count
            accumulated += turn_count

        if accumulated < target_turns * 0.8:
            raise ValueError(
                f"Planned chunks cover only {accumulated} turns (target {target_turns}). "
                "Ask the planner to cover more."
            )
        generation["chunks"] = normalised_chunks
        generation.setdefault("global_requirements", [])

        return {"scenario": scenario_block, "generation_plan": generation}

    def _parse_chunk_turns(
        self,
        payload: Dict[str, Any],
        chunk: ChunkPlan,
        scenario: Scenario,
    ) -> List[ScenarioTurn]:
        turns_raw = payload.get("turns")
        if not isinstance(turns_raw, list) or not turns_raw:
            raise ValueError("Chunk response missing 'turns' array.")

        threads = {thread.thread_id for thread in scenario.threads}
        parsed: List[ScenarioTurn] = []
        for idx, item in enumerate(turns_raw, start=1):
            thread_id = str(item.get("thread_id", "")).strip()
            if not thread_id:
                raise ValueError(f"Turn {idx} missing 'thread_id'.")
            if thread_id not in threads:
                raise ValueError(f"Turn references unknown thread_id '{thread_id}'.")
            user_message = str(item.get("user_message", "")).strip()
            if not user_message:
                raise ValueError(f"Turn {idx} missing 'user_message'.")
            expected_focus = str(item.get("expected_assistant_focus", "")).strip()
            if not expected_focus:
                raise ValueError(f"Turn {idx} missing 'expected_assistant_focus'.")
            timestamp_hint = item.get("timestamp_hint")
            if isinstance(timestamp_hint, str):
                timestamp_hint = timestamp_hint.strip() or None
            else:
                timestamp_hint = None
            escalation = item.get("escalation_trigger")
            if isinstance(escalation, str):
                escalation = escalation.strip() or None
            else:
                escalation = None
            parsed.append(
                ScenarioTurn(
                    turn_id="",
                    thread_id=thread_id,
                    user_message=user_message,
                    expected_assistant_focus=expected_focus,
                    timestamp_hint=timestamp_hint,
                    escalation_trigger=escalation,
                )
            )

        if not chunk.allow_early_termination and len(parsed) != chunk.turn_count:
            raise ValueError(
                f"Chunk {chunk.chunk_id} expected {chunk.turn_count} turns, received {len(parsed)}."
            )
        return parsed

    def _render_chunk_prompt(self, chunk: ChunkPlan, scenario: Scenario) -> str:
        scenario_meta = "\n".join(
            [
                f"Title: {scenario.title}",
                f"Background: {scenario.background}",
                f"User profile: {scenario.user_profile}",
                f"Assistant profile: {scenario.assistant_profile}",
            ]
        )
        thread_lines = []
        for thread in scenario.threads:
            entities = ", ".join(thread.key_entities) if thread.key_entities else "无"
            thread_lines.append(
                f"- {thread.thread_id} | priority={thread.priority} | intent={thread.intent} | entities={entities}"
            )
        thread_catalog = "\n".join(thread_lines)

        directive_payload = {
            "chunk_id": chunk.chunk_id,
            "turn_target": chunk.turn_count,
            "start_turn_index": chunk.start_turn_index,
            "focus_threads": chunk.focus_threads,
            "user_goals": chunk.user_goals,
            "escalations": chunk.escalations,
            "notes": chunk.notes,
            "allow_early_termination": chunk.allow_early_termination,
        }
        chunk_directive = json.dumps(directive_payload, ensure_ascii=False, indent=2)

        global_reqs = json.dumps(
            self.plan_payload["generation_plan"].get("global_requirements", []),
            ensure_ascii=False,
            indent=2,
        )
        excerpt = self._format_transcript_excerpt(scenario.dialogue, self.context_turns)
        if not excerpt:
            excerpt = "(no prior turns; this is the opening chunk)"

        return (
            self.chunk_prompt_template.replace("<<SCENARIO_METADATA>>", scenario_meta)
            .replace("<<THREAD_CATALOG>>", thread_catalog)
            .replace("<<CHUNK_DIRECTIVE>>", chunk_directive)
            .replace("<<TRANSCRIPT_EXCERPT>>", excerpt)
            .replace("<<GLOBAL_REQUIREMENTS>>", global_reqs)
        )

    def _render_metadata_prompt(self, scenario: Scenario) -> str:
        catalog_lines = []
        for thread in scenario.threads:
            entities = ", ".join(thread.key_entities) if thread.key_entities else "无"
            catalog_lines.append(
                f"- {thread.thread_id}: {thread.intent} (priority={thread.priority}, entities={entities})"
            )
        transcript_digest = self._format_transcript_digest(scenario.dialogue, limit=150)
        scenario_summary = "\n".join(
            [
                f"Title: {scenario.title}",
                f"Background: {scenario.background}",
                f"User profile: {scenario.user_profile}",
                f"Assistant profile: {scenario.assistant_profile}",
                f"Total turns: {len(scenario.dialogue)}",
            ]
        )
        return (
            self.metadata_prompt_template.replace("<<SCENARIO_SUMMARY>>", scenario_summary)
            .replace("<<THREAD_CATALOG>>", "\n".join(catalog_lines))
            .replace("<<TRANSCRIPT_DIGEST>>", transcript_digest)
        )

    @staticmethod
    def _format_transcript_excerpt(turns: Sequence[ScenarioTurn], limit: int) -> str:
        if not turns or limit <= 0:
            return ""
        recent = list(turns)[-limit:]
        lines = []
        for turn in recent:
            lines.append(f"{turn.turn_id or '?'} [{turn.thread_id}] {turn.user_message}")
        return "\n".join(lines)

    @staticmethod
    def _format_transcript_digest(turns: Sequence[ScenarioTurn], limit: Optional[int] = None) -> str:
        if not turns:
            return "(empty transcript)"
        selected: Iterable[ScenarioTurn]
        if limit is not None and len(turns) > limit:
            head = turns[: limit // 2]
            tail = turns[-limit // 2 :]
            selected = list(head) + ["..."] + list(tail)  # type: ignore
        else:
            selected = turns

        lines: List[str] = []
        for turn in selected:
            if isinstance(turn, str):
                lines.append("... (truncated) ...")
                continue
            lines.append(f"{turn.turn_id or '?'} [{turn.thread_id}] {turn.user_message}")
        return "\n".join(lines)


def main() -> None:
    args = parse_args()
    builder = ScenarioBuilder(args)
    builder.run()


if __name__ == "__main__":
    main()
