import argparse
import json
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

        return cls(
            operation=op,
            primary_branch=primary,
            target_parent_branch=target_parent,
            secondary_branches=_normalize_list("secondary_branches"),
            context_branches=_normalize_list("context_branches"),
            archive_targets=_normalize_list("archive_targets"),
            note=raw.get("note", ""),
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

    def __init__(self, root_summary: str = "Conversation starting point"):
        self.root = Node(id="root", summary=root_summary)
        self.nodes: Dict[str, Node] = {"root": self.root}
        self.children_map: Dict[str, List[str]] = {}
        self.operation_log: List[Dict[str, Any]] = []

    # --- Persistence ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_summary": self.root.summary,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "children_map": self.children_map,
            "operation_log": self.operation_log,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ContextTree":
        root_summary = payload.get("root_summary", "Conversation starting point")
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
                print("ℹ️ Label '{}' resolved to branch '...{}'.".format(reference, target[-6:]))
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

    def get_context_bundle(self, branch_ids: List[str]) -> List[Dict[str, str]]:
        """Aggregate context from several branches while keeping provenance visible."""
        bundle: List[Dict[str, str]] = []
        seen_messages: Set[str] = set()
        for branch_id in branch_ids:
            if branch_id not in self.nodes:
                continue
            summary_path = self.get_branch_summary_path(branch_id)
            tag = f"[Branch {branch_id[-6:]} Summary Chain] {summary_path}"
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
                        f"⚠️ {label} '{node_id}' does not exist; using closest known branch '{fallback}' instead."
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

        merge_primary_recovered: Optional[str] = None
        missing_merge_primary = False
        if operation == TreeOperationType.MERGE and not primary:
            # Attempt to recover a sensible anchor from secondary branches first.
            for index, candidate in enumerate(list(secondary)):
                if candidate and candidate != "root":
                    primary = candidate
                    merge_primary_recovered = candidate
                    secondary.pop(index)
                    break
            # Fall back to explicit target parent if provided and usable.
            if not primary and target_parent and target_parent != "root":
                recovered = _ensure_exists(target_parent, "target_parent_branch")
                if recovered and recovered != "root":
                    primary = recovered
                    merge_primary_recovered = recovered
            # As a last resort, reuse the latest active leaf to avoid crashing.
            if not primary:
                latest = self.get_latest_active_leaf()
                if latest and latest != "root":
                    primary = latest
                    merge_primary_recovered = latest
            if not primary:
                missing_merge_primary = True

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
        merge_downgrade_reason: Optional[str] = None
        if operation == TreeOperationType.MERGE:
            if not primary:
                if missing_merge_primary:
                    effective_operation = TreeOperationType.NEW_BRANCH
                    merge_downgrade_reason = "missing_primary_branch"
                else:
                    raise ValueError("Merge operation requires primary_branch as anchor.")
            elif len(secondary) < 1:
                effective_operation = TreeOperationType.APPEND
                merge_downgrade_reason = "no_secondary_branches"

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
            latest_leaf = self.get_latest_active_leaf()
            if latest_leaf and latest_leaf != "root":
                context_ids = [latest_leaf]
            else:
                context_ids = ["root"]
        elif context_ids == ["root"]:
            latest_leaf = self.get_latest_active_leaf()
            if latest_leaf and latest_leaf != "root":
                context_ids = [latest_leaf]

        metadata: Dict[str, Any] = {}
        if decision.note:
            metadata["decision_note"] = decision.note
        if merge_downgrade_reason:
            metadata["merge_downgraded_reason"] = merge_downgrade_reason
        if merge_primary_recovered:
            metadata["merge_primary_recovered"] = merge_primary_recovered
        if effective_operation == TreeOperationType.MERGE:
            metadata["merged_from"] = json.dumps([primary, *secondary], ensure_ascii=False)
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

    # --- Diagnostics ------------------------------------------------------------------
    def display_tree(self) -> None:
        print("🌳 Conversation tree (status: active/archived):")
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
        summary = node.summary or "(no summary)"
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
        print("Dialogue manager started. Root node created.")
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
                        f"Model {model} failed (attempt {attempt}). Last error: {exc}. Prompt digest: {digest}"
                    ) from exc
        raise RuntimeError(f"Model {model} failed: {last_error}")  # pragma: no cover

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
            raw_decision = self._call_llm(
                messages, model=self.config.cm_model, json_mode=True, phase="controller"
            )
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
        return self._call_llm(messages, model=self.config.task_model, phase="task")

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
        summary = self._call_llm(messages, model=self.config.summarizer_model, phase="summary")
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
        print(f"🧠 LLM-CM decision: {decision}")

        plan = self.tree.plan_operation(decision)
        print(f"   └─ Planning result: operation={plan.operation.value}, parent=...{plan.parent_id[-6:]}")
        if self.observer:
            self.observer.on_decision(decision, plan)

        context_messages = self.tree.get_context_bundle(plan.context_branch_ids)
        print(f"📚 Context branches aggregated: {len(plan.context_branch_ids)}")

        print("🚀 Calling LLM-Task to generate a reply...")
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


# --- 4. CLI entry ----------------------------------------------------------------------
def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    config = AppConfig.from_args(args)
    manager = DialogueManager(config)
    print("\nWelcome to BranchMind V6 — enhanced for cross-branch reasoning and atomic tree operations.")
    print("Type 'exit' to quit.")
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

#   - 扩展对话摘要，允许 80 字以内覆盖行动、截止时间与承诺，确保分支摘要携带足够信号供控制器与后续合并判定使用。
#     branchmind/main.py:1037
#   - 在控制器决策阶段注入线程标签、活跃分支快照（摘要链、最近问答、标签）并放宽提示，引导模型优先复用匹配分支。
#     branchmind/main.py:937-1010
#   - 决策回退时若只剩 root 上下文，自动补入最近活跃叶子，避免 Task 模型在空上下文下回答。branchmind/main.py:694-703
#   - handle_request 先解析线程标签并传入控制器，再将其写入新节点，保持线程一致性。branchmind/main.py:1049-1060
#   - 提示词改为英文