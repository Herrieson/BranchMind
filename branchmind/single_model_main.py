import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import AzureOpenAI

from branchmind.main import (
    ContextTree,
    DialogueObserver,
    LLMCallRecord,
    PlannedOperation,
    TreeDecision,
    TreeOperationType,
    TreeSnapshot,
    _env_or_default,
    _require_value,
    DEFAULT_API_VERSION,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_DELAY,
    DEFAULT_STATE_FILE,
)


load_dotenv()


DEFAULT_MODEL_NAME = "gpt-4o"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BranchMind — single-model dialogue manager."
    )
    parser.add_argument("--api-key", help="Azure OpenAI API key override.")
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint override.")
    parser.add_argument("--api-version", help="Azure OpenAI API version.")
    parser.add_argument("--model", help="Model name used for every stage.")
    parser.add_argument(
        "--state-path",
        help="Path to persist conversation state (JSON). Use empty string to disable.",
    )
    parser.add_argument(
        "--retry-attempts",
        type=int,
        help=f"Retry attempts for LLM calls (default {DEFAULT_RETRY_ATTEMPTS}).",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        help=f"Seconds to wait between LLM retries (default {DEFAULT_RETRY_DELAY}).",
    )
    return parser


@dataclass
class SingleModelAppConfig:
    api_key: str
    azure_endpoint: str
    api_version: str
    model: str
    state_path: Optional[Path]
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_delay: float = DEFAULT_RETRY_DELAY

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SingleModelAppConfig":
        api_key = args.api_key or _env_or_default("API_KEY")
        azure_endpoint = args.azure_endpoint or _env_or_default("AZURE_ENDPOINT")
        api_version = args.api_version or _env_or_default("API_VERSION", DEFAULT_API_VERSION)
        model = (
            args.model
            or _env_or_default("MODEL_NAME")
            or _env_or_default("MODEL")
            or _env_or_default("TASK_MODEL")
            or _env_or_default("CM_MODEL")
            or _env_or_default("SUMMARIZER_MODEL")
            or DEFAULT_MODEL_NAME
        )

        state_path_str = args.state_path
        if state_path_str is None:
            state_path_str = _env_or_default("STATE_PATH", DEFAULT_STATE_FILE)
        if state_path_str is not None and not state_path_str.strip():
            state_path = None
        else:
            state_path = Path(state_path_str) if state_path_str else None

        retry_attempts = (
            args.retry_attempts
            if args.retry_attempts is not None
            else int(_env_or_default("LLM_RETRY_ATTEMPTS", str(DEFAULT_RETRY_ATTEMPTS)))
        )
        retry_delay = (
            args.retry_delay
            if args.retry_delay is not None
            else float(_env_or_default("LLM_RETRY_DELAY", str(DEFAULT_RETRY_DELAY)))
        )

        return cls(
            api_key=_require_value("API_KEY", api_key),
            azure_endpoint=_require_value("AZURE_ENDPOINT", azure_endpoint),
            api_version=_require_value("API_VERSION", api_version),
            model=_require_value("MODEL_NAME", model),
            state_path=state_path,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
        )


class SingleModelDialogueManager:
    def __init__(self, config: SingleModelAppConfig, observer: Optional[DialogueObserver] = None):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.azure_endpoint,
        )
        self.retry_attempts = max(1, config.retry_attempts)
        self.retry_delay = max(0.0, config.retry_delay)
        self.tree = self._load_state() or ContextTree()
        self.observer = observer
        print("Single-model dialogue manager started. Root node ready.")
        if self.config.state_path:
            print(f"State file: {self.config.state_path.resolve()}")

    def _load_state(self) -> Optional[ContextTree]:
        if not self.config.state_path:
            return None
        path = self.config.state_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tree = ContextTree.from_dict(data)
            print("Conversation tree restored from state file.")
            return tree
        except Exception as exc:
            print(f"⚠️ Failed to load state file '{path}': {exc}")
            return None

    def save_state(self) -> None:
        if not self.config.state_path:
            return
        path = self.config.state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.tree.to_dict()
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"⚠️ Error while saving state file: {exc}")

    def set_observer(self, observer: Optional[DialogueObserver]) -> None:
        self.observer = observer

    @staticmethod
    def _prompt_digest(messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        for message in reversed(messages):
            if message.get("role") == "user":
                snippet = message.get("content", "")
                break
        else:
            snippet = messages[-1].get("content", "")
        snippet = snippet.replace("\n", " ").strip()
        return snippet[:160]

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        *,
        json_mode: bool = False,
        phase: str = "generic",
    ) -> str:
        start_time = time.perf_counter()
        response_kwargs: Dict[str, Any] = {"model": self.config.model, "messages": messages}
        if json_mode:
            response_kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self.client.chat.completions.create(**response_kwargs)
                content = response.choices[0].message.content
                if content is None:
                    raise RuntimeError("LLM returned empty content.")
                latency = time.perf_counter() - start_time
                usage_payload = {
                    "prompt_tokens": getattr(getattr(response, "usage", None), "prompt_tokens", 0),
                    "completion_tokens": getattr(getattr(response, "usage", None), "completion_tokens", 0),
                    "total_tokens": getattr(getattr(response, "usage", None), "total_tokens", 0),
                }
                if self.observer:
                    record = LLMCallRecord(
                        phase=phase,
                        model=self.config.model,
                        latency=latency,
                        usage=usage_payload,
                        metadata={"json_mode": json_mode},
                    )
                    self.observer.on_llm_call(record)
                return content
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_attempts:
                    time.sleep(self.retry_delay)
                else:
                    digest = self._prompt_digest(messages)
                    raise RuntimeError(
                        f"Model {self.config.model} failed (attempt {attempt}). Last error: {exc}. Prompt digest: {digest}"
                    ) from exc
        raise RuntimeError(f"Model {self.config.model} failed: {last_error}")  # pragma: no cover

    def _default_decision(self) -> TreeDecision:
        latest_leaf = self.tree.get_latest_active_leaf()
        if latest_leaf:
            return TreeDecision(
                operation=TreeOperationType.APPEND,
                primary_branch=latest_leaf,
                note="fallback: append to latest active leaf",
            )
        return TreeDecision(
            operation=TreeOperationType.NEW_BRANCH,
            note="fallback: create new branch",
        )

    def llm_cm_decide(self, query: str, thread_tag: Optional[str] = None) -> TreeDecision:
        active_branches = self.tree.get_active_branches()
        branch_full_paths = {
            branch_id: self.tree.get_branch_summary_path(branch_id)
            for branch_id in active_branches.keys()
        }
        thread_hint = thread_tag if thread_tag else "no explicit thread tag"

        def _trim(text: Optional[str], limit: int = 160) -> str:
            if not text:
                return ""
            cleaned = text.strip()
            return cleaned if len(cleaned) <= limit else f"{cleaned[:limit]}…"

        branch_profiles: List[Dict[str, Any]] = []
        for branch_id in active_branches.keys():
            try:
                node = self.tree.get_node(branch_id)
            except KeyError:
                continue
            branch_profiles.append(
                {
                    "branch_id": branch_id,
                    "summary_path": branch_full_paths.get(branch_id, ""),
                    "latest_summary": _trim(node.summary, 120),
                    "latest_query": _trim(node.query, 160),
                    "latest_response": _trim(node.response, 160),
                    "tags": self.tree._collect_tags(branch_id),
                }
            )
        tag_index = {
            tag: {
                "branch_id": branch_id,
                "summary_path": branch_full_paths.get(branch_id, self.tree.get_branch_summary_path(branch_id)),
            }
            for tag, branch_id in self.tree.get_tag_map().items()
        }

        prompt = f"""
You are a professional dialogue context scheduler. In a system with advanced tree operations, choose the best strategy for the latest user request.

The system supports the following operations:
1. "new"      — Create a brand-new branch under the root.
2. "append"   — Append the current query to the tail of the specified branch.
3. "merge"    — Combine knowledge from multiple branches to craft a new answer; the merged branches are usually archived afterwards.
4. "split"    — When a branch becomes overloaded, record it and start a new child branch from its parent (you may choose an earlier ancestor or the root if needed).
5. "archive"  — Mark old branches as archived so they no longer influence future decisions, while remaining available for citation.

Output strictly formatted JSON:
{{
  "operation": "new" | "append" | "merge" | "split" | "archive",
  "primary_branch": "<primary branch ID or null>",
  "target_parent_branch": "<for split overrides, provide ancestor branch ID or null>",
  "secondary_branches": ["<branch IDs referenced by merge or split>", ...],
  "context_branches": ["<branch IDs whose context must be aggregated>", ...],
  "archive_targets": ["<branch IDs that should be archived immediately>", ...],
  "note": "Short note for human debugging"
}}
The value of target_parent_branch must be an ancestor of primary_branch (or "root"). Use null if no override is required.

Current request thread tag: {thread_hint}

Snapshot of active branches (active leaves only):
{json.dumps(branch_profiles, indent=2, ensure_ascii=False)}

Summary paths for active branches:
{json.dumps(branch_full_paths, indent=2, ensure_ascii=False)}

Available tag index (tag -> branch info):
{json.dumps(tag_index, indent=2, ensure_ascii=False)}

Latest user query: "{query}"

Maintain contextual continuity based on the thread tag and label index: append when either points to an existing branch; use merge when multiple branches must be integrated (listing them explicitly); only create new for genuinely new topics.
"""
        messages = [
            {"role": "system", "content": "You are a tree-structured context scheduler AI. Output JSON only."},
            {"role": "user", "content": prompt},
        ]

        try:
            raw_decision = self._call_llm(messages, json_mode=True, phase="controller")
            decision_payload = json.loads(raw_decision)
            return TreeDecision.from_raw(decision_payload, self.tree.root.id)
        except Exception as exc:
            print(f"⚠️ Controller decision failed; using fallback strategy: {exc}")
            return self._default_decision()

    def llm_task_execute(self, query: str, context: List[Dict[str, str]]) -> str:
        system_message = (
            "You are a capable AI assistant. Use the provided context (possibly from multiple branches) "
            "to craft a clear, accurate answer that cites relevant sources."
        )
        messages = [{"role": "system", "content": system_message}] + context + [
            {"role": "user", "content": query}
        ]
        return self._call_llm(messages, phase="task")

    def _summarize_interaction(self, query: str, response: str) -> str:
        prompt = (
            "Create a concise summary (no more than 80 words) covering key actions, deadlines, commitments, "
            "and next steps for the following exchange:\n\n"
            f"User asked: {query}\nAI answered: {response}"
        )
        messages = [
            {"role": "system", "content": "You are a text summarization expert. Output only a concise summary."},
            {"role": "user", "content": prompt},
        ]
        summary = self._call_llm(messages, phase="summary")
        return summary.strip().replace("\n", " ")

    def handle_request(self, query: str, *, thread_id: Optional[str] = None) -> str:
        print("\n" + "=" * 60)
        print(f"Received new request: {query}")

        thread_tag: Optional[str]
        if isinstance(thread_id, str):
            candidate = thread_id.strip()
            thread_tag = candidate if candidate.startswith("T-") else None
        else:
            thread_tag = None

        decision = self.llm_cm_decide(query, thread_tag=thread_tag)
        print(f"🧠 LLM decision: {decision}")

        plan = self.tree.plan_operation(decision)
        print(f"   └─ Planning result: operation={plan.operation.value}, parent=...{plan.parent_id[-6:]}")
        if self.observer:
            self.observer.on_decision(decision, plan)

        context_messages = self.tree.get_context_bundle(plan.context_branch_ids)
        print(f"📚 Context branches aggregated: {len(plan.context_branch_ids)}")

        print("🚀 Calling model to generate a reply...")
        response = self.llm_task_execute(query, context_messages)

        print("📝 Generating interaction summary...")
        summary = self._summarize_interaction(query, response)

        new_node = self.tree.commit_operation(plan, query, response, summary, thread_tag=thread_tag)
        print(f"   └─ New node '...{new_node.id[-6:]}' created with summary: '{summary}'")
        if plan.archive_targets:
            print(f"   └─ Archived branches: {', '.join('...'+bid[-6:] for bid in plan.archive_targets)}")

        if self.observer:
            snapshot = self.tree.compute_metrics(plan.operation)
            self.observer.on_tree_update(snapshot, new_node)

        self.tree.display_tree()
        self.save_state()
        print("=" * 60 + "\n")
        return response


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = SingleModelAppConfig.from_args(args)
    manager = SingleModelDialogueManager(config)
    print("\nWelcome to BranchMind (single-model). Type 'exit' to quit.")
    print("-" * 20)
    try:
        while True:
            user_query = input("You: ").strip()
            if user_query.lower() == "exit":
                break
            if not user_query:
                continue
            try:
                ai_response = manager.handle_request(user_query)
                print(f"AI: {ai_response}")
            except Exception as exc:
                print(f"Error while handling request: {exc}")
    finally:
        manager.save_state()
        print("State saved. Goodbye.")


if __name__ == "__main__":
    main()
