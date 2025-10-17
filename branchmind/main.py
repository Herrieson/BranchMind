import argparse
import json
import math
import os
import re
import time
import uuid
from difflib import get_close_matches
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import AzureOpenAI


# --- 1. Configuration ------------------------------------------------------------------
load_dotenv()


DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_CM_MODEL = "gpt-4o-mini"
DEFAULT_TASK_MODEL = "gpt-4o"
DEFAULT_SUMMARIZER_MODEL = "gpt-4o-mini"
DEFAULT_STATE_FILE = "branchmind_state.json"
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 2.0


def _env_or_default(name: str, fallback: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is not None and value.strip():
        return value
    return fallback


def _require_value(name: str, value: Optional[str]) -> str:
    if not value:
        raise EnvironmentError(f"Missing required configuration value for '{name}'.")
    return value


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BranchMind V6 (improved) — tree-structured dialogue manager."
    )
    parser.add_argument("--api-key", help="Azure OpenAI API key override.")
    parser.add_argument("--azure-endpoint", help="Azure OpenAI endpoint override.")
    parser.add_argument("--api-version", help="Azure OpenAI API version.")
    parser.add_argument("--cm-model", help="Controller model name.")
    parser.add_argument("--task-model", help="Task model name.")
    parser.add_argument("--summarizer-model", help="Summarizer model name.")
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
class AppConfig:
    api_key: str
    azure_endpoint: str
    api_version: str
    cm_model: str
    task_model: str
    summarizer_model: str
    state_path: Optional[Path]
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS
    retry_delay: float = DEFAULT_RETRY_DELAY

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "AppConfig":
        api_key = args.api_key or _env_or_default("API_KEY")
        azure_endpoint = args.azure_endpoint or _env_or_default("AZURE_ENDPOINT")
        api_version = args.api_version or _env_or_default("API_VERSION", DEFAULT_API_VERSION)
        cm_model = args.cm_model or _env_or_default("CM_MODEL", DEFAULT_CM_MODEL)
        task_model = args.task_model or _env_or_default("TASK_MODEL", DEFAULT_TASK_MODEL)
        summarizer_model = args.summarizer_model or _env_or_default(
            "SUMMARIZER_MODEL", DEFAULT_SUMMARIZER_MODEL
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
            cm_model=_require_value("CM_MODEL", cm_model),
            task_model=_require_value("TASK_MODEL", task_model),
            summarizer_model=_require_value("SUMMARIZER_MODEL", summarizer_model),
            state_path=state_path,
            retry_attempts=retry_attempts,
            retry_delay=retry_delay,
        )


@dataclass
class LLMCallRecord:
    """Telemetry entry describing a single LLM invocation."""

    phase: str
    model: str
    latency: float
    usage: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreeSnapshot:
    total_nodes: int
    active_nodes: int
    active_branches: int
    max_depth: int
    last_node_id: str
    last_operation: str


class DialogueObserver:
    """Interface for stress-test instrumentation hooks."""

    def on_llm_call(self, record: LLMCallRecord) -> None:  # pragma: no cover - interface hook
        pass

    def on_decision(self, decision: "TreeDecision", plan: "PlannedOperation") -> None:  # pragma: no cover
        pass

    def on_tree_update(self, snapshot: TreeSnapshot, node: "Node") -> None:  # pragma: no cover
        pass


class AdaptationMonitor:
    """Tracks controller confidence and produces a smooth degradation coefficient."""

    EVENT_WEIGHTS = {
        "controller_failure": 0.4,
        "plan_failure": 0.3,
        "final_retry": 0.2,
        "operation_simplified": 0.15,
    }

    def __init__(self, *, decay_factor: float = 0.35, recovery_rate: float = 0.08, min_confidence: float = 0.05):
        self.confidence = 1.0
        self.decay_factor = max(0.0, min(1.0, decay_factor))
        self.recovery_rate = max(0.0, min(1.0, recovery_rate))
        self.min_confidence = max(0.0, min(0.5, min_confidence))
        self._round_penalty = 0.0

    def start_round(self) -> None:
        self._round_penalty = 0.0

    def record_event(self, name: str, weight: Optional[float] = None) -> None:
        event_weight = weight if weight is not None else self.EVENT_WEIGHTS.get(name, 0.1)
        if event_weight <= 0:
            return
        self._round_penalty = min(1.5, self._round_penalty + event_weight)

    def finish_round(self) -> None:
        penalty = self._round_penalty
        if penalty > 0:
            drop = penalty * self.decay_factor
            self.confidence = max(self.min_confidence, self.confidence - drop)
        else:
            recovery = self.recovery_rate * (1.0 - self.confidence)
            self.confidence = min(1.0, self.confidence + recovery)
        self._round_penalty = 0.0

    @property
    def alpha(self) -> float:
        scaled = (self.confidence - 0.5) * 4.0
        return 1.0 / (1.0 + math.exp(-scaled))


class SlidingMemory:
    """Lightweight sliding window memory used when BranchMind confidence drops."""

    def __init__(self, *, window_turns: int = 4, summary_tail: int = 3):
        self.window_turns = max(1, window_turns)
        self.summary_tail = max(0, summary_tail)
        self.system_prompt = (
            "你是一个勤勉助手，主要依赖最近的对话窗口和摘要完成任务。"
            "在信息不足时，保持行动建议具体、简洁。"
        )
        self.dialogue: List[Dict[str, str]] = []
        self.summaries: List[str] = []

    def record(self, query: str, response: str, summary: str) -> None:
        if query:
            self.dialogue.append({"role": "user", "content": query})
        if response:
            self.dialogue.append({"role": "assistant", "content": response})
        cleaned_summary = (summary or "").strip()
        if cleaned_summary:
            self.summaries.append(cleaned_summary)
        max_messages = max(10, self.window_turns * 6)
        if len(self.dialogue) > max_messages:
            self.dialogue = self.dialogue[-max_messages:]
        if len(self.summaries) > 12:
            self.summaries = self.summaries[-12:]

    def build_prompt(self) -> List[Dict[str, str]]:
        if not self.dialogue:
            return []
        window_messages = self.dialogue[-(self.window_turns * 2) :]
        messages: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt}]
        if self.summary_tail and self.summaries:
            tail = " | ".join(self.summaries[-self.summary_tail :])
            if tail:
                messages.append({"role": "system", "content": f"[滑窗摘要] {tail}"})
        messages.extend(window_messages)
        return messages


# --- 2. Tree primitives ----------------------------------------------------------------
class TreeOperationType(str, Enum):
    APPEND = "append"
    NEW_BRANCH = "new"
    MERGE = "merge"
    SPLIT = "split"
    ARCHIVE = "archive"


@dataclass
class TreeDecision:
    """Structured view of the controller model decision."""

    operation: TreeOperationType
    primary_branch: Optional[str] = None
    target_parent_branch: Optional[str] = None
    secondary_branches: List[str] = field(default_factory=list)
    context_branches: List[str] = field(default_factory=list)
    archive_targets: List[str] = field(default_factory=list)
    note: str = ""
    requires_final_synthesis: bool = False

    @classmethod
    def from_raw(cls, raw: Dict[str, Any], root_id: str) -> "TreeDecision":
        try:
            op = TreeOperationType(raw.get("operation", "").lower())
        except ValueError as exc:
            raise ValueError(f"Unsupported operation: {raw}") from exc

        def _normalize_scalar(value: Optional[str]) -> Optional[str]:
            if value is None:
                return None
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"", "null", "none"}:
                    return None
            return value

        primary = _normalize_scalar(raw.get("primary_branch"))
        if primary == "root":
            primary = root_id

        def _normalize_list(key: str) -> List[str]:
            values = raw.get(key) or []
            normalized: List[str] = []
            for val in values:
                normalized_val = _normalize_scalar(val)
                if not normalized_val:
                    continue
                normalized.append(normalized_val if normalized_val != "root" else root_id)
            return normalized

        target_parent = _normalize_scalar(raw.get("target_parent_branch"))
        if target_parent == "root":
            target_parent = root_id

        def _normalize_bool(key: str) -> bool:
            value = raw.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y"}:
                    return True
                if lowered in {"false", "0", "no", "n"}:
                    return False
            return False

        return cls(
            operation=op,
            primary_branch=primary,
            target_parent_branch=target_parent,
            secondary_branches=_normalize_list("secondary_branches"),
            context_branches=_normalize_list("context_branches"),
            archive_targets=_normalize_list("archive_targets"),
            note=raw.get("note", ""),
            requires_final_synthesis=_normalize_bool("requires_final_synthesis"),
        )


@dataclass
class PlannedOperation:
    """Concrete plan derived from a TreeDecision before the task model is called."""

    operation: TreeOperationType
    parent_id: str
    context_branch_ids: List[str]
    archive_targets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Node:
    """A single interaction inside the context tree."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    query: str = ""
    response: str = ""
    summary: str = ""
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "query": self.query,
            "response": self.response,
            "summary": self.summary,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Node":
        return cls(
            id=payload.get("id", str(uuid.uuid4())),
            parent_id=payload.get("parent_id"),
            query=payload.get("query", ""),
            response=payload.get("response", ""),
            summary=payload.get("summary", ""),
            status=payload.get("status", "active"),
            metadata=payload.get("metadata", {}) or {},
        )


class ContextTree:
    """Holds the conversation graph and exposes atomic operations on it."""

    TAG_PATTERN = re.compile(r"T-[^\s,，。；;:：()（）<>《》【】\\]+")
    TAG_STRIP_CHARS = ",，。.;；:：()（）<>《》【】[]{}“”\"'`"

    def __init__(self, root_summary: str = "对话的起点"):
        self.root = Node(id="root", summary=root_summary)
        self.nodes: Dict[str, Node] = {"root": self.root}
        self.children_map: Dict[str, List[str]] = {}
        self.operation_log: List[Dict[str, Any]] = []
        self.snapshots: Dict[str, Dict[str, str]] = {}

    # --- Persistence ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_summary": self.root.summary,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "children_map": self.children_map,
            "operation_log": self.operation_log,
            "snapshots": self.snapshots,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ContextTree":
        root_summary = payload.get("root_summary", "对话的起点")
        tree = cls(root_summary=root_summary)
        nodes_payload: Dict[str, Any] = payload.get("nodes", {})
        tree.nodes = {}
        for node_id, node_data in nodes_payload.items():
            node = Node.from_dict(node_data)
            tree.nodes[node_id] = node
            if node_id == "root":
                tree.root = node
        if "root" not in tree.nodes:
            tree.root = Node(id="root", summary=root_summary)
            tree.nodes["root"] = tree.root

        raw_children = payload.get("children_map", {})
        tree.children_map = {}
        for parent_id, children in raw_children.items():
            valid_children = [cid for cid in children if cid in tree.nodes]
            if valid_children:
                tree.children_map[parent_id] = valid_children

        # Ensure child lists reflect actual parent references.
        for node in tree.nodes.values():
            if node.parent_id:
                tree.children_map.setdefault(node.parent_id, [])
                if node.id not in tree.children_map[node.parent_id]:
                    tree.children_map[node.parent_id].append(node.id)

        tree.operation_log = list(payload.get("operation_log", []))
        tree.snapshots = {
            key: value
            for key, value in payload.get("snapshots", {}).items()
            if isinstance(value, dict)
        }
        return tree

    @classmethod
    def _extract_tags_from_text(cls, text: str) -> List[str]:
        if not text:
            return []
        matches = cls.TAG_PATTERN.findall(text)
        tags: List[str] = []
        for raw in matches:
            cleaned = raw.strip(cls.TAG_STRIP_CHARS).strip()
            if cleaned and cleaned not in tags:
                tags.append(cleaned)
        return tags

    def _get_node_tags(self, node_id: Optional[str]) -> List[str]:
        if not node_id or node_id not in self.nodes:
            return []
        node = self.nodes[node_id]
        raw = node.metadata.get("tags", [])
        collected: List[str] = []
        if isinstance(raw, list):
            collected.extend(tag for tag in raw if isinstance(tag, str) and tag)
        if isinstance(raw, str) and raw:
            # tolerate legacy string payloads by splitting on commas
            try:
                decoded = json.loads(raw)
                if isinstance(decoded, list):
                    collected.extend(tag for tag in decoded if isinstance(tag, str) and tag)
            except json.JSONDecodeError:
                pass
            collected.extend(item.strip() for item in raw.split(",") if item.strip())

        if not collected:
            fallback_sources = [
                node.query,
                node.summary,
                node.metadata.get("decision_note", "") if isinstance(node.metadata, dict) else "",
            ]
            for source in fallback_sources:
                for tag in self._extract_tags_from_text(source or ""):
                    if tag not in collected:
                        collected.append(tag)

        return collected

    def _collect_tags(self, node_id: Optional[str]) -> List[str]:
        seen: List[str] = []
        current_id = node_id
        while current_id and current_id in self.nodes:
            for tag in self._get_node_tags(current_id):
                if tag not in seen:
                    seen.append(tag)
            current = self.nodes[current_id]
            current_id = current.parent_id
        return seen

    def _build_tag_map(self, include_archived: bool = False) -> Dict[str, str]:
        order = {node_id: index for index, node_id in enumerate(self.nodes.keys())}
        if include_archived:
            branch_ids = [node_id for node_id in self.nodes.keys() if node_id != "root"]
        else:
            branch_ids = list(self.get_active_branches().keys())
        tag_map: Dict[str, str] = {}
        for branch_id in branch_ids:
            for tag in self._collect_tags(branch_id):
                if not tag:
                    continue
                existing = tag_map.get(tag)
                if not existing or order.get(branch_id, -1) >= order.get(existing, -1):
                    tag_map[tag] = branch_id
        return tag_map

    def get_tag_map(self, include_archived: bool = False) -> Dict[str, str]:
        return self._build_tag_map(include_archived=include_archived)

    def resolve_branch_reference(self, reference: Optional[str]) -> Optional[str]:
        if reference is None:
            return None
        reference = reference.strip()
        if not reference:
            return None
        if reference in self.nodes:
            return reference
        if reference.startswith("T-"):
            tag_map = self._build_tag_map(include_archived=False)
            target = tag_map.get(reference)
            if not target:
                tag_map = self._build_tag_map(include_archived=True)
                target = tag_map.get(reference)
            if target:
                print(f"ℹ️ 标签 '{reference}' 解析到分支 '...{target[-6:]}'.")
                return target
        return reference

    # --- Node utilities ---------------------------------------------------------------
    def get_node(self, node_id: str) -> Node:
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' does not exist.")
        return self.nodes[node_id]

    def add_node(self, node: Node) -> None:
        if node.parent_id not in self.nodes:
            raise ValueError(f"Parent '{node.parent_id}' does not exist.")
        self.nodes[node.id] = node
        self.children_map.setdefault(node.parent_id, []).append(node.id)

    def is_ancestor(self, ancestor_id: str, node_id: str) -> bool:
        if node_id not in self.nodes:
            return False
        if ancestor_id == "root":
            return True
        if ancestor_id not in self.nodes:
            return False
        current = self.nodes[node_id]
        while current.parent_id:
            if current.parent_id == ancestor_id:
                return True
            current = self.nodes.get(current.parent_id)
            if current is None:
                break
        return False

    def _descendants(self, node_id: str) -> Set[str]:
        queue = [node_id]
        descendants: Set[str] = set()
        while queue:
            current = queue.pop()
            children = self.children_map.get(current, [])
            for child in children:
                descendants.add(child)
                queue.append(child)
        return descendants

    def _node_depth(self, node_id: str) -> int:
        depth = 0
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            depth += 1
            current = self.nodes.get(current.parent_id)
        return depth

    def archive_branch(self, branch_id: str) -> None:
        for node_id in {branch_id} | self._descendants(branch_id):
            if node_id in self.nodes:
                self.nodes[node_id].status = "archived"

    def get_context_path(self, leaf_id: str) -> List[Dict[str, str]]:
        if leaf_id not in self.nodes:
            return []
        path: List[Dict[str, str]] = []
        curr = self.nodes[leaf_id]
        while curr.id != "root":
            path.insert(0, {"role": "assistant", "content": curr.response})
            path.insert(0, {"role": "user", "content": curr.query})
            if curr.parent_id is None:
                break
            curr = self.nodes[curr.parent_id]
        return path

    def get_branch_summary_path(self, leaf_id: str) -> str:
        if leaf_id not in self.nodes:
            return ""
        summaries: List[str] = []
        curr = self.nodes[leaf_id]
        while curr is not None:
            summaries.insert(0, curr.summary)
            if curr.parent_id is None:
                break
            curr = self.nodes.get(curr.parent_id)
        return " -> ".join(filter(None, summaries))

    def get_active_branches(self) -> Dict[str, str]:
        leaves: Dict[str, str] = {}
        for node_id, node in self.nodes.items():
            if node_id == "root" or node.status != "active":
                continue
            children = self.children_map.get(node_id, [])
            has_active_child = any(self.nodes[ch].status == "active" for ch in children)
            if not has_active_child:
                leaves[node_id] = node.summary
        return leaves

    def get_latest_active_leaf(self) -> Optional[str]:
        active_leaves = self.get_active_branches()
        if not active_leaves:
            return None
        for node_id in reversed(list(self.nodes.keys())):
            if node_id in active_leaves:
                return node_id
        return None

    def _recent_active_leaves(self) -> List[str]:
        active = self.get_active_branches()
        if not active:
            return []
        ordered: List[str] = []
        seen: Set[str] = set()
        for entry in reversed(self.operation_log):
            node_id = str(entry.get("node_id", ""))
            if node_id in active and node_id not in seen:
                ordered.append(node_id)
                seen.add(node_id)
        for node_id in active:
            if node_id not in seen:
                ordered.append(node_id)
        return ordered

    def _suggest_merge_secondaries(self, primary: str, limit: int = 3) -> List[str]:
        suggestions: List[str] = []
        for candidate in self._recent_active_leaves():
            if candidate == primary:
                continue
            node = self.nodes.get(candidate)
            if not node or node.status != "active":
                continue
            suggestions.append(candidate)
            if len(suggestions) >= limit:
                break
        return suggestions

    def get_context_bundle(self, branch_ids: List[str]) -> List[Dict[str, str]]:
        """Aggregate context from several branches while keeping provenance visible."""
        bundle: List[Dict[str, str]] = []
        seen_messages: Set[str] = set()
        for branch_id in branch_ids:
            if branch_id not in self.nodes:
                continue
            summary_path = self.get_branch_summary_path(branch_id)
            tag = f"[分支 {branch_id[-6:]} 摘要链] {summary_path}"
            if tag not in seen_messages:
                bundle.append({"role": "system", "content": tag})
                seen_messages.add(tag)
            branch_messages = self.get_context_path(branch_id)
            for index, message in enumerate(branch_messages):
                key = f"{branch_id}::{index}::{message['role']}::{message['content']}"
                if key in seen_messages:
                    continue
                bundle.append(message)
                seen_messages.add(key)
        return bundle

    def compute_metrics(self, last_operation: TreeOperationType) -> TreeSnapshot:
        total_nodes = len(self.nodes)
        active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")
        active_branches = len(self.get_active_branches())
        max_depth = self._compute_max_depth()
        last_node_id = ""
        if self.operation_log:
            last_entry = self.operation_log[-1]
            last_node_id = str(last_entry.get("node_id", ""))
        return TreeSnapshot(
            total_nodes=total_nodes,
            active_nodes=active_nodes,
            active_branches=active_branches,
            max_depth=max_depth,
            last_node_id=last_node_id,
            last_operation=last_operation.value,
        )

    def _compute_max_depth(self) -> int:
        if not self.nodes:
            return 0
        max_depth = 0
        stack = [(self.root.id, 0)]
        while stack:
            node_id, depth = stack.pop()
            max_depth = max(max_depth, depth)
            for child in self.children_map.get(node_id, []):
                stack.append((child, depth + 1))
        return max_depth

    # --- Planning / Commit ------------------------------------------------------------
    def plan_operation(self, decision: TreeDecision) -> PlannedOperation:
        """Validate an operation and compute the parent/context set."""
        operation = decision.operation

        def _ensure_exists(node_id: Optional[str], label: str) -> Optional[str]:
            if node_id is None:
                return None
            if node_id not in self.nodes:
                candidates = get_close_matches(node_id, list(self.nodes.keys()), n=1, cutoff=0.75)
                if candidates:
                    fallback = candidates[0]
                    print(
                        f"⚠️ {label} '{node_id}' 不存在，自动改用最接近的已知分支 '{fallback}'."
                    )
                    return fallback
                raise ValueError(f"{label} '{node_id}' does not exist in tree.")
            return node_id

        primary = self.resolve_branch_reference(decision.primary_branch)
        if operation == TreeOperationType.NEW_BRANCH:
            if primary and primary not in self.nodes:
                primary = None
        else:
            primary = _ensure_exists(primary, "primary_branch")

        target_parent = self.resolve_branch_reference(decision.target_parent_branch)
        if operation == TreeOperationType.SPLIT:
            target_parent = _ensure_exists(target_parent, "target_parent_branch")
        elif target_parent and target_parent not in self.nodes:
            target_parent = None

        resolved_secondaries = [
            self.resolve_branch_reference(b) for b in decision.secondary_branches
        ]
        if operation in (TreeOperationType.MERGE, TreeOperationType.SPLIT):
            secondary = []
            for candidate in resolved_secondaries:
                normalized = _ensure_exists(candidate, "secondary_branch")
                if normalized:
                    secondary.append(normalized)
        else:
            secondary = [b for b in resolved_secondaries if b and b in self.nodes]

        context_ids: List[str] = []
        resolved_contexts = [self.resolve_branch_reference(b) for b in decision.context_branches]
        for candidate in [primary, *secondary, *resolved_contexts]:
            if not candidate or candidate not in self.nodes:
                continue
            if candidate not in context_ids:
                context_ids.append(candidate)

        resolved_archives = [
            self.resolve_branch_reference(branch_id) for branch_id in decision.archive_targets
        ]
        archive_targets = []
        for branch_id in resolved_archives:
            normalized = _ensure_exists(branch_id, "archive_target")
            if normalized:
                archive_targets.append(normalized)

        split_parent_hint: Optional[str] = None

        effective_operation = operation
        auto_filled_secondaries: List[str] = []
        if operation == TreeOperationType.MERGE:
            if not primary:
                raise ValueError("Merge operation requires primary_branch as anchor.")
            if len(secondary) < 1:
                auto_filled_secondaries = self._suggest_merge_secondaries(primary)
                if auto_filled_secondaries:
                    for candidate in auto_filled_secondaries:
                        if candidate not in secondary:
                            secondary.append(candidate)
                            if candidate not in context_ids:
                                context_ids.append(candidate)
                else:
                    raise ValueError(
                        "Merge operation requires at least one secondary branch; "
                        "controller did not provide any and no active branches are available."
                    )

        if effective_operation == TreeOperationType.NEW_BRANCH:
            parent_id = "root"
        elif effective_operation == TreeOperationType.APPEND:
            if not primary:
                raise ValueError("Append operation requires primary_branch.")
            parent_id = primary
        elif effective_operation == TreeOperationType.MERGE:
            if len(secondary) < 1:
                raise ValueError("Merge operation requires at least one secondary branch.")
            parent_id = "root"
            archives: Set[Optional[str]] = set(archive_targets)
            archives.update(secondary)
            archives.add(primary)
            archive_targets = [bid for bid in archives if bid and bid != "root"]
        elif effective_operation == TreeOperationType.SPLIT:
            if not primary:
                raise ValueError("Split operation requires primary_branch to split from.")
            parent_node = self.get_node(primary)
            split_parent_hint = parent_node.parent_id or "root"
            if target_parent:
                if not self.is_ancestor(target_parent, primary):
                    raise ValueError(
                        "Split target_parent_branch must be an ancestor of primary_branch."
                    )
                parent_id = target_parent
            else:
                parent_id = split_parent_hint
            if primary and primary not in context_ids:
                context_ids.append(primary)
        elif effective_operation == TreeOperationType.ARCHIVE:
            if not archive_targets:
                raise ValueError("Archive operation must specify archive_targets.")
            parent_id = "root"
        else:
            raise ValueError(f"Unsupported operation type: {effective_operation}")

        if not context_ids:
            context_ids = ["root"]

        metadata: Dict[str, Any] = {}
        if decision.note:
            metadata["decision_note"] = decision.note
        if effective_operation == TreeOperationType.MERGE:
            metadata["merged_from"] = json.dumps([primary, *secondary], ensure_ascii=False)
            if auto_filled_secondaries:
                metadata["auto_secondary_branches"] = auto_filled_secondaries
        if effective_operation == TreeOperationType.SPLIT:
            metadata["split_from"] = primary or ""
            if split_parent_hint:
                metadata["split_parent_hint"] = split_parent_hint
            if target_parent and target_parent != split_parent_hint:
                metadata["split_parent_override"] = target_parent

        return PlannedOperation(
            operation=effective_operation,
            parent_id=parent_id,
            context_branch_ids=context_ids,
            archive_targets=[bid for bid in archive_targets if bid],
            metadata=metadata,
        )

    def _classify_snapshots(self, query: str, response: str) -> Dict[str, str]:
        payload: Dict[str, str] = {}
        q_lower = query.lower()
        r_lower = response.lower()
        if any(keyword in q_lower for keyword in ["主日历", "时间线", "日程", "calendar"]) or any(
            keyword in r_lower for keyword in ["主日历", "时间线", "日程", "calendar"]
        ):
            payload["calendar"] = response
        if any(keyword in q_lower for keyword in ["预算", "台账", "现金流"]) or any(
            keyword in r_lower for keyword in ["预算", "台账", "现金流"]
        ):
            payload["budget"] = response
        if "ics" in q_lower or "ics" in r_lower:
            payload["ics"] = response
        if any(keyword in q_lower for keyword in ["风险", "冲突"]) or any(
            keyword in r_lower for keyword in ["风险", "冲突"]
        ):
            payload["risk"] = response
        return payload

    def _update_snapshot(self, category: str, node_id: str, content: str) -> None:
        snippet = content.strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000] + " ..."
        self.snapshots[category] = {"node_id": node_id, "content": snippet}

    def _should_flag_autosplit(self, branch_id: Optional[str]) -> bool:
        if not branch_id or branch_id not in self.nodes or branch_id == "root":
            return False
        tag_count = len(self._collect_tags(branch_id))
        depth = self._node_depth(branch_id)
        return tag_count >= 8 or depth >= 4

    def commit_operation(
        self,
        plan: PlannedOperation,
        query: str,
        response: str,
        summary: str,
        thread_tag: Optional[str] = None,
    ) -> Node:
        parent_tags = self._collect_tags(plan.parent_id)
        tags_from_query = self._extract_tags_from_text(query)
        computed_tags: List[str] = []
        for tag in [*parent_tags, *tags_from_query]:
            if tag and tag not in computed_tags:
                computed_tags.append(tag)

        if thread_tag and thread_tag not in computed_tags:
            computed_tags.append(thread_tag)

        merged_from = plan.metadata.get("merged_from")
        if merged_from:
            try:
                merged_ids = json.loads(merged_from)
            except (json.JSONDecodeError, TypeError):
                merged_ids = []
            if isinstance(merged_ids, list):
                for branch_id in merged_ids:
                    for tag in self._collect_tags(branch_id):
                        if tag and tag not in computed_tags:
                            computed_tags.append(tag)

        split_from = plan.metadata.get("split_from")
        if split_from:
            for tag in self._collect_tags(split_from):
                if tag and tag not in computed_tags:
                    computed_tags.append(tag)

        new_node = Node(
            parent_id=plan.parent_id,
            query=query,
            response=response,
            summary=summary.strip(),
            metadata=plan.metadata.copy(),
        )
        if computed_tags:
            new_node.metadata["tags"] = computed_tags

        self.add_node(new_node)

        for category, content in self._classify_snapshots(query, response).items():
            self._update_snapshot(category, new_node.id, content)

        if plan.operation == TreeOperationType.APPEND and self._should_flag_autosplit(plan.parent_id):
            branch_id = plan.parent_id
            overload_message = (
                f"分支 ...{branch_id[-6:]} 标签/深度已过载，请优先执行 SPLIT 或重组。"
            )
            self._update_snapshot("autosplit", branch_id, overload_message)

        for branch_id in plan.archive_targets:
            self.archive_branch(branch_id)

        self.operation_log.append(
            {
                "op": plan.operation.value,
                "node_id": new_node.id,
                "metadata": plan.metadata,
                "archives": plan.archive_targets,
            }
        )
        return new_node

    def get_snapshot_messages(self, categories: Optional[List[str]] = None) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if categories is None:
            items = self.snapshots.items()
        else:
            items = ((category, self.snapshots.get(category)) for category in categories)
        for category, payload in items:
            if not payload:
                continue
            content = payload.get("content", "")
            if not content:
                continue
            label = category.upper()
            messages.append({"role": "system", "content": f"[Snapshot:{label}] {content}"})
        return messages

    # --- Diagnostics ------------------------------------------------------------------
    def display_tree(self) -> None:
        print("🌳 对话树状结构图 (status: active/archived):")
        active_ids = {bid for bid, _ in self.get_active_branches().items()}
        self._display_node_recursive(self.root.id, "", True, active_ids)

    def _display_node_recursive(
        self, node_id: str, prefix: str, is_last: bool, active_ids: Set[str]
    ) -> None:
        node = self.get_node(node_id)
        is_active = node_id in active_ids or node.status == "active"
        status_marker = (
            "🟢" if is_active and node.status == "active" else "⚪️" if node.status == "active" else "⚫️"
        )
        connector = "└── " if is_last else "├── "
        summary = node.summary or "(无摘要)"
        note = ""
        if node.metadata.get("merged_from"):
            note = " ⤳ merge"
        elif node.metadata.get("split_from"):
            note = " ⤳ split"
        print(f"{prefix}{connector}{status_marker} {summary} (ID: ...{node_id[-6:]}){note}")
        children = self.children_map.get(node_id, [])
        new_prefix = prefix + ("    " if is_last else "│   ")
        for i, child_id in enumerate(children):
            self._display_node_recursive(child_id, new_prefix, i == len(children) - 1, active_ids)


# --- 3. Dialogue manager ---------------------------------------------------------------
class DialogueManager:
    FINAL_CHECKLIST_FIELDS = [
        "integrated_summary",
        "action_plan",
        "resource_alignment",
        "risks",
        "open_questions",
    ]
    FINAL_CHECKLIST_LABELS = {
        "integrated_summary": "integrated summary",
        "action_plan": "action plan",
        "resource_alignment": "resource alignment",
        "risks": "risk coverage",
        "open_questions": "open questions",
    }
    FINAL_CHECKLIST_PATTERN = re.compile(
        r"FinalChecklist\s*:?\s*```json\s*({.*?})\s*```",
        re.IGNORECASE | re.DOTALL,
    )

    def __init__(self, config: AppConfig, observer: Optional[DialogueObserver] = None):
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
        self.adaptation = AdaptationMonitor()
        self.sliding_memory = SlidingMemory()
        self._pending_operation: Optional[TreeDecision] = None
        print("对话管理器已启动。根节点已创建。")
        if self.config.state_path:
            print(f"状态文件: {self.config.state_path.resolve()}")

    def _load_state(self) -> Optional[ContextTree]:
        if not self.config.state_path:
            return None
        path = self.config.state_path
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            tree = ContextTree.from_dict(data)
            print("已从持久化文件恢复对话树。")
            return tree
        except Exception as exc:
            print(f"⚠️ 无法加载状态文件 '{path}': {exc}")
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
            print(f"⚠️ 保存状态文件时出错: {exc}")

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
        model: str,
        *,
        json_mode: bool = False,
        phase: str = "generic",
    ) -> str:
        start_time = time.perf_counter()
        response_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
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
                        model=model,
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
                        f"调用模型 {model} 失败（尝试 {attempt} 次）。最后错误：{exc}. 提示摘要: {digest}"
                    ) from exc
        raise RuntimeError(f"调用模型 {model} 失败: {last_error}")  # pragma: no cover

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

    def _requires_final_synthesis(self, _query: str, decision: TreeDecision) -> bool:
        return (
            decision.operation == TreeOperationType.MERGE
            and decision.requires_final_synthesis
        )

    @classmethod
    def _parse_final_checklist(cls, text: str) -> Optional[Dict[str, str]]:
        match = cls.FINAL_CHECKLIST_PATTERN.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None

        checklist: Dict[str, str] = {}
        for field in cls.FINAL_CHECKLIST_FIELDS:
            value = data.get(field, "")
            if isinstance(value, str):
                normalized = value.strip().lower()
            elif isinstance(value, bool):
                normalized = "done" if value else "missing"
            else:
                normalized = str(value).strip().lower()
            checklist[field] = normalized
        return checklist

    @staticmethod
    def _final_synthesis_guardrail() -> str:
        return (
            "【终局综合指引】请生成一份已经汇总所有上下文的最终答复，满足以下要求：\n"
            "1. 给出合并后的整体概览与关键背景。\n"
            "2. 列出可执行的行动计划（包括时间或责任人，如有）。\n"
            "3. 说明资源/依赖的落实情况，并指出缺口。\n"
            "4. 识别主要风险及缓解方案。\n"
            "5. 总结仍待确认或待决的问题。\n"
            "在回答末尾添加标题 `FinalChecklist` 并紧随一个 JSON 代码块，字段取值只能是 \"done\" 或 \"missing\"：\n"
            "```json\n"
            "{\n"
            '  "integrated_summary": "done",\n'
            '  "action_plan": "done",\n'
            '  "resource_alignment": "missing",\n'
            '  "risks": "done",\n'
            '  "open_questions": "missing"\n'
            "}\n"
            "```\n"
            "若任何信息不足，请在主体中说明并在 JSON 中标记为 \"missing\"。"
        )

    def _evaluate_final_response(self, text: str) -> List[str]:
        checklist = self._parse_final_checklist(text)
        if checklist is None:
            return ["FinalChecklist JSON 缺失或无法解析"]

        missing: List[str] = []
        for field in self.FINAL_CHECKLIST_FIELDS:
            status = checklist.get(field, "")
            if status != "done":
                missing.append(self.FINAL_CHECKLIST_LABELS.get(field, field))
        return missing

    @staticmethod
    def _final_synthesis_retry_prompt(missing: List[str]) -> str:
        checklist = "、".join(missing)
        return (
            "【终局补充提醒】刚才的输出未满足以下检查项："
            f"{checklist}。请完善主体内容，并保证回答末尾提供结构化的 FinalChecklist JSON。"
        )

    @staticmethod
    def _clone_plan(plan: PlannedOperation, **overrides: Any) -> PlannedOperation:
        payload = {
            "operation": plan.operation,
            "parent_id": plan.parent_id,
            "context_branch_ids": list(plan.context_branch_ids),
            "archive_targets": list(plan.archive_targets),
            "metadata": dict(plan.metadata),
        }
        payload.update(overrides)
        return PlannedOperation(**payload)

    def _resolve_branch_for_append(self, *candidates: Optional[str]) -> str:
        for candidate in candidates:
            if candidate and candidate in self.tree.nodes and candidate != "root":
                return candidate
        latest = self.tree.get_latest_active_leaf()
        if latest and latest in self.tree.nodes:
            return latest
        return "root"

    def _adapt_plan(self, decision: TreeDecision, plan: PlannedOperation, alpha: float) -> PlannedOperation:
        adjusted = self._clone_plan(plan)
        simplified = False
        if decision.operation == TreeOperationType.MERGE and alpha < 0.55:
            target = self._resolve_branch_for_append(decision.primary_branch, adjusted.parent_id)
            if target == "root":
                adjusted = self._clone_plan(
                    adjusted,
                    operation=TreeOperationType.NEW_BRANCH,
                    parent_id="root",
                    context_branch_ids=["root"],
                    archive_targets=[],
                )
            else:
                adjusted = self._clone_plan(
                    adjusted,
                    operation=TreeOperationType.APPEND,
                    parent_id=target,
                    context_branch_ids=[target],
                    archive_targets=[],
                )
            adjusted.metadata.setdefault("degraded_from", "merge")
            adjusted.metadata["degraded_to"] = adjusted.operation.value
            simplified = True
        elif decision.operation == TreeOperationType.SPLIT and alpha < 0.5:
            target = self._resolve_branch_for_append(decision.primary_branch, adjusted.parent_id)
            if target == "root":
                adjusted = self._clone_plan(
                    adjusted,
                    operation=TreeOperationType.NEW_BRANCH,
                    parent_id="root",
                    context_branch_ids=["root"],
                    archive_targets=[],
                )
            else:
                adjusted = self._clone_plan(
                    adjusted,
                    operation=TreeOperationType.APPEND,
                    parent_id=target,
                    context_branch_ids=[target],
                    archive_targets=[],
                )
            adjusted.metadata.setdefault("degraded_from", "split")
            adjusted.metadata["degraded_to"] = adjusted.operation.value
            simplified = True
        elif decision.operation == TreeOperationType.ARCHIVE and alpha < 0.4 and adjusted.archive_targets:
            adjusted = self._clone_plan(
                adjusted,
                archive_targets=[],
            )
            adjusted.metadata.setdefault("degraded_from", "archive")
            adjusted.metadata["degraded_to"] = "skip"
            simplified = True

        if simplified:
            self.adaptation.record_event("operation_simplified")

        context_ids = list(dict.fromkeys(adjusted.context_branch_ids))
        if not context_ids:
            context_ids = ["root"]
        max_keep = max(1, int(round(max(alpha, 0.05) * len(context_ids))))
        trimmed = context_ids[:max_keep]
        if "root" not in trimmed:
            trimmed.insert(0, "root")
        adjusted.context_branch_ids = trimmed
        return adjusted

    @staticmethod
    def _blend_contexts(
        tree_context: List[Dict[str, str]], window_context: List[Dict[str, str]], alpha: float
    ) -> List[Dict[str, str]]:
        if not window_context:
            return tree_context
        if alpha <= 0.05:
            return window_context
        if alpha >= 0.95:
            return tree_context

        blended: List[Dict[str, str]] = []
        keep_count = int(round(alpha * len(tree_context)))
        if tree_context:
            keep_count = max(1, keep_count)
        blended.extend(tree_context[:keep_count])

        seen = {(msg["role"], msg["content"]) for msg in blended}
        for index, message in enumerate(window_context):
            if message["role"] == "system" and index == 0 and alpha > 0.4:
                continue
            key = (message["role"], message["content"])
            if key in seen:
                continue
            blended.append(message)
            seen.add(key)
        return blended

    def llm_cm_decide(self, query: str) -> TreeDecision:
        active_branches = self.tree.get_active_branches()
        branch_full_paths = {
            branch_id: self.tree.get_branch_summary_path(branch_id)
            for branch_id in active_branches.keys()
        }
        tag_index = {
            tag: {
                "branch_id": branch_id,
                "summary_path": branch_full_paths.get(branch_id, self.tree.get_branch_summary_path(branch_id)),
            }
            for tag, branch_id in self.tree.get_tag_map().items()
        }

        prompt = f"""
你是一个专业的对话上下文调度器，需要在一个支持高级树操作的系统中为最新用户提问选择策略。

系统支持的操作：
1. "new"      —— 在 root 下创建一个全新的分支。
2. "append"   —— 将当前提问附加到指定分支的尾部。
3. "merge"    —— 将多个分支的知识合并，生成新的回答；通常需要归档被合并的旧分支。
4. "split"    —— 如果某个分支主题过载，先记录分支，再从其父节点开启一个新的子分支；如有需要，可指定更早的祖先或 root 作为新节点的父节点。
5. "archive"  —— 将若干旧分支标记为归档，不再参与后续决策，但回答当前问题时仍可引用。

如果用户期望的是一次终局性合并（需要严格的终局检查/一体化交付），请将 `requires_final_synthesis` 设置为 true，否则为 false。

请输出严格的 JSON：
{{
  "operation": "new" | "append" | "merge" | "split" | "archive",
  "primary_branch": "<主要分支ID或null>",
  "target_parent_branch": "<当 operation 为 split 且需要覆盖挂载位置时，填写祖先分支ID或null>",
  "secondary_branches": ["<被merge或split参考的分支ID>", ...],
  "context_branches": ["<需要参与上下文聚合的分支ID>", ...],
  "archive_targets": ["<需要立即归档的分支ID>", ...],
  "requires_final_synthesis": true | false,
  "note": "用于人类调试的简短说明"
}}
其中 target_parent_branch 必须是 primary_branch 的祖先（或 "root"），若无需覆盖请填写 null。
若选择 "merge"，primary_branch 必须填写一个非 null 的活跃分支 ID，且该分支将作为合并锚点。

当前活跃分支的摘要路径：
{json.dumps(branch_full_paths, indent=2, ensure_ascii=False)}

可用标签索引（tag -> branch信息）：
{json.dumps(tag_index, indent=2, ensure_ascii=False)}

用户最新提问："{query}"
"""
        messages = [
            {"role": "system", "content": "你是树状上下文的调度AI，只能输出JSON。"},
            {"role": "user", "content": prompt},
        ]

        try:
            raw_decision = self._call_llm(
                messages, model=self.config.cm_model, json_mode=True, phase="controller"
            )
            decision_payload = json.loads(raw_decision)
            decision = TreeDecision.from_raw(decision_payload, self.tree.root.id)
            if decision.operation == TreeOperationType.MERGE and not decision.primary_branch:
                fallback_anchor = self.tree.get_latest_active_leaf()
                if fallback_anchor:
                    print(
                        f"⚠️ 控制模型未提供 merge 主分支，自动使用最新活跃分支 '...{fallback_anchor[-6:]}' 作为锚点。"
                    )
                    decision.primary_branch = fallback_anchor
                else:
                    print(
                        "⚠️ 控制模型返回 merge 但没有可用的主分支，已改为在 root 下创建新分支。"
                    )
                    decision.operation = TreeOperationType.NEW_BRANCH
            return decision
        except Exception as exc:
            print(f"⚠️ 控制模型决策失败，将使用回退策略：{exc}")
            self.adaptation.record_event("controller_failure")
            return self._default_decision()

    def llm_task_execute(self, query: str, context: List[Dict[str, str]]) -> str:
        system_message = (
            "你是一个能干的AI助手，请根据提供的上下文（可能来自多个分支），"
            "生成清晰、准确且能引用相关上下文来源的回答。"
        )
        messages = [{"role": "system", "content": system_message}] + context + [
            {"role": "user", "content": query}
        ]
        return self._call_llm(messages, model=self.config.task_model, phase="task")

    def _summarize_interaction(self, query: str, response: str) -> str:
        prompt = (
            "请为以下问答生成紧凑摘要（不超过20个字），重点描述行动或结论：\n\n"
            f"用户问：{query}\nAI答：{response}"
        )
        messages = [
            {"role": "system", "content": "你是一个文本摘要专家，只输出简洁摘要。"},
            {"role": "user", "content": prompt},
        ]
        summary = self._call_llm(messages, model=self.config.summarizer_model, phase="summary")
        return summary.strip().replace("\n", " ")

    def handle_request(self, query: str, *, thread_id: Optional[str] = None) -> str:
        print("\n" + "=" * 60)
        print(f"接收到新请求: {query}")

        self.adaptation.start_round()
        alpha = self.adaptation.alpha
        print(f"🧭 控制置信度: {self.adaptation.confidence:.2f}，自适应系数 α={alpha:.2f}")

        response: str = ""
        try:
            decision = self.llm_cm_decide(query)
            print(f"🧠 LLM-CM 决策: {decision}")

            if decision.operation in (TreeOperationType.MERGE, TreeOperationType.SPLIT):
                self._pending_operation = decision
            else:
                self._pending_operation = None

            try:
                plan = self.tree.plan_operation(decision)
            except ValueError as exc:
                print(f"⚠️ 规划失败，将退回默认策略：{exc}")
                self.adaptation.record_event("plan_failure")
                fallback_decision = self._default_decision()
                plan = self.tree.plan_operation(fallback_decision)
                decision = fallback_decision
                self._pending_operation = None

            plan = self._adapt_plan(decision, plan, alpha)
            print(f"   └─ 规划结果：operation={plan.operation.value}, parent=...{plan.parent_id[-6:]}")

            if self.observer:
                self.observer.on_decision(decision, plan)

            base_context_messages = self.tree.get_context_bundle(plan.context_branch_ids)
            window_context_messages = self.sliding_memory.build_prompt()
            blended_context = self._blend_contexts(base_context_messages, window_context_messages, alpha)
            print(f"📚 聚合上下文分支数量: {len(plan.context_branch_ids)}，滑窗补充={len(window_context_messages)}")

            requires_final = self._requires_final_synthesis(query, decision)
            guardrail_prompt: Optional[str] = None
            snapshot_messages: List[Dict[str, str]] = []
            if requires_final:
                snapshot_messages = self.tree.get_snapshot_messages()
                guardrail_prompt = self._final_synthesis_guardrail()
                context_messages = [{"role": "system", "content": guardrail_prompt}]
                context_messages.extend(snapshot_messages)
                context_messages.extend(blended_context)
            else:
                context_messages = blended_context

            print("🚀 正在调用 LLM-Task 生成回复...")
            response = self.llm_task_execute(query, context_messages)

            if requires_final:
                missing = self._evaluate_final_response(response)
                if missing:
                    self.adaptation.record_event("final_retry")
                    print(f"⚠️ 终局检查缺项：{', '.join(missing)}，将触发补充生成。")
                    retry_prompt = self._final_synthesis_retry_prompt(missing)
                    retry_context = [{"role": "system", "content": retry_prompt}]
                    if guardrail_prompt:
                        retry_context.append({"role": "system", "content": guardrail_prompt})
                    retry_context.extend(snapshot_messages)
                    retry_context.extend(blended_context)
                    response = self.llm_task_execute(query, retry_context)

            print("📝 正在生成交互摘要...")
            summary = self._summarize_interaction(query, response)

            if isinstance(thread_id, str):
                candidate = thread_id.strip()
                thread_tag = candidate if candidate.startswith("T-") else None
            else:
                thread_tag = None

            new_node = self.tree.commit_operation(plan, query, response, summary, thread_tag=thread_tag)
            print(f"   └─ 新节点 '...{new_node.id[-6:]}' 已创建，摘要为: '{summary}'")
            if plan.archive_targets:
                print(f"   └─ 已归档分支: {', '.join('...'+bid[-6:] for bid in plan.archive_targets)}")

            if self.observer:
                snapshot = self.tree.compute_metrics(plan.operation)
                self.observer.on_tree_update(snapshot, new_node)

            self.sliding_memory.record(query, response, summary)

            self.tree.display_tree()
            self.save_state()
            print("=" * 60 + "\n")
            return response
        finally:
            self._pending_operation = None
            self.adaptation.finish_round()


# --- 4. CLI entry ----------------------------------------------------------------------
def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = AppConfig.from_args(args)
    manager = DialogueManager(config)
    print("\n欢迎来到 BranchMind V6 —— 支持跨分支推理与原子树操作的改进版。")
    print("输入 'exit' 退出。")
    print("-" * 20)
    try:
        while True:
            user_query = input("你: ").strip()
            if user_query.lower() == "exit":
                break
            if not user_query:
                continue
            try:
                ai_response = manager.handle_request(user_query)
                print(f"AI: {ai_response}")
            except Exception as exc:
                print(f"处理请求时出错: {exc}")
    finally:
        manager.save_state()
        print("状态已保存，再见。")


if __name__ == "__main__":
    main()
