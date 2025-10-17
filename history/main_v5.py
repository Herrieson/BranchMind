import os
import uuid
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from openai import AzureOpenAI


# --- 1. Configuration ------------------------------------------------------------------
load_dotenv()


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {name}")
    return value


client = AzureOpenAI(
    api_key=_require_env("API_KEY"),
    api_version="2024-12-01-preview",
    azure_endpoint=_require_env("AZURE_ENDPOINT"),
)

CM_MODEL = "gpt-4o-mini"
TASK_MODEL = "gpt-4o"
SUMMARIZER_MODEL = "gpt-4o-mini"


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
    def from_raw(cls, raw: Dict, root_id: str) -> "TreeDecision":
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
    """The concrete plan derived from a TreeDecision before the task model is called."""

    operation: TreeOperationType
    parent_id: str
    context_branch_ids: List[str]
    archive_targets: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class Node:
    """A single interaction inside the context tree."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    query: str = ""
    response: str = ""
    summary: str = ""
    status: str = "active"
    metadata: Dict[str, str] = field(default_factory=dict)


class ContextTree:
    """Holds the conversation graph and exposes atomic operations on it."""

    def __init__(self, root_summary: str = "对话的起点"):
        self.root = Node(id="root", summary=root_summary)
        self.nodes: Dict[str, Node] = {"root": self.root}
        self.children_map: Dict[str, List[str]] = {}
        self.operation_log: List[Dict] = []

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
        if ancestor_id == "root":
            return True
        if ancestor_id not in self.nodes or node_id not in self.nodes:
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
        return " -> ".join(summaries)

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
            for message in self.get_context_path(branch_id):
                key = f"{message['role']}::{message['content']}"
                if key in seen_messages:
                    continue
                bundle.append(message)
                seen_messages.add(key)
        return bundle

    # --- Planning / Commit ------------------------------------------------------------
    def plan_operation(self, decision: TreeDecision) -> PlannedOperation:
        """Validate an operation and compute the parent/context set."""
        operation = decision.operation

        def _ensure_exists(node_id: Optional[str], label: str) -> Optional[str]:
            if node_id is None:
                return None
            if node_id not in self.nodes:
                raise ValueError(f"{label} '{node_id}' does not exist in tree.")
            return node_id

        primary = decision.primary_branch
        if operation == TreeOperationType.NEW_BRANCH:
            # Some controller outputs may optimistically assign a placeholder branch ID.
            # For a pure new-branch operation the parent is always the root, so we safely ignore it.
            if primary and primary not in self.nodes:
                primary = None
        else:
            primary = _ensure_exists(primary, "primary_branch")
        target_parent = decision.target_parent_branch
        if operation == TreeOperationType.SPLIT:
            target_parent = _ensure_exists(target_parent, "target_parent_branch")
        elif target_parent and target_parent not in self.nodes:
            target_parent = None

        if operation in (TreeOperationType.MERGE, TreeOperationType.SPLIT):
            secondary = [
                _ensure_exists(b, "secondary_branch") for b in decision.secondary_branches
            ]
        else:
            secondary = [
                b for b in decision.secondary_branches if b in self.nodes
            ]
        context_ids: List[str] = []
        for candidate in [primary, *secondary, *decision.context_branches]:
            if not candidate or candidate not in self.nodes:
                continue
            if candidate not in context_ids:
                context_ids.append(candidate)

        archive_targets = [
            _ensure_exists(branch_id, "archive_target")
            for branch_id in decision.archive_targets
        ]

        split_parent_hint: Optional[str] = None

        if operation == TreeOperationType.NEW_BRANCH:
            parent_id = "root"
        elif operation == TreeOperationType.APPEND:
            if not primary:
                raise ValueError("Append operation requires primary_branch.")
            parent_id = primary
        elif operation == TreeOperationType.MERGE:
            if not primary:
                raise ValueError("Merge operation requires primary_branch as anchor.")
            if len(secondary) < 1:
                raise ValueError("Merge operation requires at least one secondary branch.")
            parent_id = "root"
            archives: Set[Optional[str]] = set(archive_targets)
            archives.update(secondary)
            archives.add(primary)
            archive_targets = [bid for bid in archives if bid and bid != "root"]
        elif operation == TreeOperationType.SPLIT:
            if not primary:
                raise ValueError("Split operation requires primary_branch to split from.")
            parent_node = self.get_node(primary)
            split_parent_hint = parent_node.parent_id or "root"
            if target_parent:
                if not self.is_ancestor(target_parent, primary):
                    raise ValueError("Split target_parent_branch must be an ancestor of primary_branch.")
                parent_id = target_parent
            else:
                parent_id = split_parent_hint
            if primary and primary not in context_ids:
                context_ids.append(primary)
        elif operation == TreeOperationType.ARCHIVE:
            if not archive_targets:
                raise ValueError("Archive operation must specify archive_targets.")
            parent_id = "root"
        else:
            raise ValueError(f"Unsupported operation type: {operation}")

        if not context_ids:
            context_ids = ["root"]

        metadata = {}
        if decision.note:
            metadata["decision_note"] = decision.note
        if operation == TreeOperationType.MERGE:
            metadata["merged_from"] = json.dumps([primary, *secondary], ensure_ascii=False)
        if operation == TreeOperationType.SPLIT:
            metadata["split_from"] = primary or ""
            if split_parent_hint:
                metadata["split_parent_hint"] = split_parent_hint
            if target_parent and target_parent != split_parent_hint:
                metadata["split_parent_override"] = target_parent

        return PlannedOperation(
            operation=operation,
            parent_id=parent_id,
            context_branch_ids=context_ids,
            archive_targets=[bid for bid in archive_targets if bid],
            metadata=metadata,
        )

    def commit_operation(self, plan: PlannedOperation, query: str, response: str, summary: str) -> Node:
        new_node = Node(
            parent_id=plan.parent_id,
            query=query,
            response=response,
            summary=summary.strip(),
            metadata=plan.metadata.copy(),
        )
        self.add_node(new_node)

        for branch_id in plan.archive_targets:
            self.archive_branch(branch_id)

        self.operation_log.append({
            "op": plan.operation.value,
            "node_id": new_node.id,
            "metadata": plan.metadata,
            "archives": plan.archive_targets,
        })
        return new_node

    # --- Diagnostics ------------------------------------------------------------------
    def display_tree(self) -> None:
        print("🌳 对话树状结构图 (status: active/archived):")
        active_ids = {bid for bid, _ in self.get_active_branches().items()}
        self._display_node_recursive(self.root.id, "", True, active_ids)

    def _display_node_recursive(self, node_id: str, prefix: str, is_last: bool, active_ids: Set[str]) -> None:
        node = self.get_node(node_id)
        is_active = node_id in active_ids or node.status == "active"
        status_marker = "🟢" if is_active and node.status == "active" else "⚪️" if node.status == "active" else "⚫️"
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
    def __init__(self):
        self.tree = ContextTree()
        print("对话管理器已启动。根节点已创建。")

    def _call_llm(self, messages: List[Dict[str, str]], model: str, json_mode: bool = False) -> str:
        response_kwargs = {"model": model, "messages": messages}
        if json_mode:
            response_kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**response_kwargs)
            return response.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"调用模型 {model} 失败: {exc}") from exc

    def llm_cm_decide(self, query: str) -> TreeDecision:
        active_branches = self.tree.get_active_branches()
        branch_full_paths = {
            branch_id: self.tree.get_branch_summary_path(branch_id)
            for branch_id in active_branches.keys()
        }

        prompt = f"""
你是一个专业的对话上下文调度器，需要在一个支持高级树操作的系统中为最新用户提问选择策略。

系统支持的操作：
1. "new"      —— 在 root 下创建一个全新的分支。
2. "append"   —— 将当前提问附加到指定分支的尾部。
3. "merge"    —— 将多个分支的知识合并，生成新的回答；通常需要归档被合并的旧分支。
4. "split"    —— 如果某个分支主题过载，先记录分支，再从其父节点开启一个新的子分支；如有需要，可指定更早的祖先或 root 作为新节点的父节点。
5. "archive"  —— 将若干旧分支标记为归档，不再参与后续决策，但回答当前问题时仍可引用。

请输出严格的 JSON：
{{
  "operation": "new" | "append" | "merge" | "split" | "archive",
  "primary_branch": "<主要分支ID或null>",
  "target_parent_branch": "<当 operation 为 split 且需要覆盖挂载位置时，填写祖先分支ID或null>",
  "secondary_branches": ["<被merge或split参考的分支ID>", ...],
  "context_branches": ["<需要参与上下文聚合的分支ID>", ...],
  "archive_targets": ["<需要立即归档的分支ID>", ...],
  "note": "用于人类调试的简短说明"
}}
其中 target_parent_branch 必须是 primary_branch 的祖先（或 \"root\"），若无需覆盖请填写 null。

当前活跃分支的摘要路径：
{json.dumps(branch_full_paths, indent=2, ensure_ascii=False)}

用户最新提问："{query}"
"""
        messages = [
            {"role": "system", "content": "你是树状上下文的调度AI，只能输出JSON。"},
            {"role": "user", "content": prompt},
        ]

        raw_decision = self._call_llm(messages, model=CM_MODEL, json_mode=True)
        try:
            decision_payload = json.loads(raw_decision)
            return TreeDecision.from_raw(decision_payload, self.tree.root.id)
        except Exception as exc:
            raise ValueError(f"无法解析CM模型输出：{raw_decision}") from exc

    def llm_task_execute(self, query: str, context: List[Dict[str, str]]) -> str:
        system_message = (
            "你是一个能干的AI助手，请根据提供的上下文（可能来自多个分支），"
            "生成清晰、准确且能引用相关上下文来源的回答。"
        )
        messages = [{"role": "system", "content": system_message}] + context + [
            {"role": "user", "content": query}
        ]
        return self._call_llm(messages, model=TASK_MODEL)

    def _summarize_interaction(self, query: str, response: str) -> str:
        prompt = (
            "请为以下问答生成紧凑摘要（不超过20个字），重点描述行动或结论：\n\n"
            f"用户问：{query}\nAI答：{response}"
        )
        messages = [
            {"role": "system", "content": "你是一个文本摘要专家，只输出简洁摘要。"},
            {"role": "user", "content": prompt},
        ]
        summary = self._call_llm(messages, model=SUMMARIZER_MODEL)
        return summary.strip().replace("\n", " ")

    def handle_request(self, query: str) -> str:
        print("\n" + "=" * 60)
        print(f"接收到新请求: {query}")

        decision = self.llm_cm_decide(query)
        print(f"🧠 LLM-CM 决策: {decision}")

        plan = self.tree.plan_operation(decision)
        print(f"   └─ 规划结果：operation={plan.operation.value}, parent=...{plan.parent_id[-6:]}")

        context_messages = self.tree.get_context_bundle(plan.context_branch_ids)
        print(f"📚 聚合上下文分支数量: {len(plan.context_branch_ids)}")

        print("🚀 正在调用 LLM-Task 生成回复...")
        response = self.llm_task_execute(query, context_messages)

        print("📝 正在生成交互摘要...")
        summary = self._summarize_interaction(query, response)

        new_node = self.tree.commit_operation(plan, query, response, summary)
        print(f"   └─ 新节点 '...{new_node.id[-6:]}' 已创建，摘要为: '{summary}'")
        if plan.archive_targets:
            print(f"   └─ 已归档分支: {', '.join('...'+bid[-6:] for bid in plan.archive_targets)}")

        self.tree.display_tree()
        print("=" * 60 + "\n")
        return response


# --- 4. CLI entry ----------------------------------------------------------------------
def main() -> None:
    manager = DialogueManager()
    print("\n欢迎来到 BranchMind V5 —— 支持跨分支推理与原子树操作的版本。")
    print("输入 'exit' 退出。")
    print("-" * 20)
    while True:
        user_query = input("你: ").strip()
        if user_query.lower() == "exit":
            break
        try:
            ai_response = manager.handle_request(user_query)
            print(f"AI: {ai_response}")
        except Exception as exc:
            print(f"处理请求时出错: {exc}")


if __name__ == "__main__":
    main()
