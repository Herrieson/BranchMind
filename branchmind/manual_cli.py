import argparse
import textwrap
from typing import Dict, List, Optional

from branchmind.main import (
    AppConfig,
    DialogueManager,
    TreeDecision,
    TreeOperationType,
    build_argument_parser,
)


_OPERATION_HELP = textwrap.dedent(
    """
    Available operations:
      • append  – continue the story on the chosen branch (default).
      • new     – start a fresh top-level branch.
      • merge   – consolidate several branches and optionally archive them.
      • split   – fork the current branch under an ancestor.
      • archive – retire one or more branches without adding replies.
    Type the keyword, press Enter for the default, or '?' to view this guide again.
    """
).strip()

_LAST_BRANCH_CHOICES: List[str] = []
_LAST_BRANCH_LIST_CACHE: Dict[str, List[str]] = {}


def _short_branch(branch_id: Optional[str]) -> str:
    if not branch_id:
        return ""
    return f"...{branch_id[-6:]}" if len(branch_id) > 6 else branch_id


def _resolve_branch_alias(token: Optional[str]) -> Optional[str]:
    if not token:
        return token
    raw = token.strip()
    if not raw:
        return None
    if raw.isdigit():
        index = int(raw)
        if 1 <= index <= len(_LAST_BRANCH_CHOICES):
            resolved = _LAST_BRANCH_CHOICES[index - 1]
            print(f"  ↳ Using selection {raw} → {_short_branch(resolved)}")
            return resolved
    if raw.startswith("#") and raw[1:].isdigit():
        index = int(raw[1:])
        if 1 <= index <= len(_LAST_BRANCH_CHOICES):
            resolved = _LAST_BRANCH_CHOICES[index - 1]
            print(f"  ↳ Using selection #{index} → {_short_branch(resolved)}")
            return resolved
    return raw


def _split_csv(value: str) -> List[str]:
    items = [item.strip() for item in value.split(",")]
    normalized: List[str] = []
    for item in items:
        if not item:
            continue
        alias = _resolve_branch_alias(item)
        if alias:
            normalized.append(alias)
    return normalized


def _ask_operation() -> TreeOperationType:
    allowed = {op.value: op for op in TreeOperationType}
    prompt = (
        "Select operation [append/new/merge/split/archive] (Enter = append | ? = help): "
    )
    while True:
        choice = input(prompt).strip().lower()
        if not choice:
            return TreeOperationType.APPEND
        if choice in {"?", "help"}:
            print("\n" + _OPERATION_HELP + "\n")
            continue
        op = allowed.get(choice)
        if op:
            return op
        print("Invalid operation. Please choose from append, new, merge, split, archive.")


def _ask_branch(
    message: str,
    *,
    default: Optional[str] = None,
    required: bool = False,
) -> Optional[str]:
    default_alias = _resolve_branch_alias(default) if default else default
    default_hint = _short_branch(default_alias) if default_alias else ""
    hint = f" [{default_hint}]" if default_hint else ""
    while True:
        raw = input(f"{message}{hint}: ").strip()
        if raw:
            resolved = _resolve_branch_alias(raw)
            if resolved:
                return resolved
            print("Unable to interpret branch reference. Try a branch number, ID, or tag.")
            continue
        if default_alias:
            print(f"  ↳ Using default {default_hint or default_alias}")
            return default_alias
        if not required:
            return None
        print("A value is required.")


def _ask_branch_list(
    message: str,
    *,
    required: bool = False,
) -> List[str]:
    cache_key = message.strip()
    previous = _LAST_BRANCH_LIST_CACHE.get(cache_key, [])
    hint_bits: List[str] = []
    if previous:
        rendered = ", ".join(_short_branch(branch_id) for branch_id in previous)
        hint_bits.append(f"Enter to reuse [{rendered}]")
    if not required:
        hint_bits.append("Enter to skip")
    hint = f" ({'; '.join(hint_bits)})" if hint_bits else ""
    while True:
        raw = input(f"{message}{hint}: ").strip()
        if not raw:
            if previous:
                print(
                    "  ↳ Reusing previous selection: "
                    + ", ".join(_short_branch(branch_id) for branch_id in previous)
                )
                return list(previous)
            if required:
                print("At least one branch must be provided.")
                continue
            return []
        selection = _split_csv(raw)
        if not selection:
            print("Unable to interpret selection. Use comma-separated branch numbers, IDs, or tags.")
            continue
        _LAST_BRANCH_LIST_CACHE[cache_key] = selection
        return selection


def _ask_optional_note() -> str:
    raw = input("Optional note (supports %s placeholder, Enter to skip): ").strip()
    return raw


def _describe_active_tree(manager: DialogueManager, *, latest_hint: Optional[str] = None) -> None:
    tree = manager.tree
    print("\n📊 Current conversation tree snapshot:")
    tree.display_tree()
    active = tree.get_active_branches()
    branch_items = list(active.items())
    global _LAST_BRANCH_CHOICES
    _LAST_BRANCH_CHOICES = [branch_id for branch_id, _ in branch_items]
    if branch_items:
        print("\n🔥 Active branches:")
        for index, (branch_id, summary) in enumerate(branch_items, start=1):
            marker = "★" if latest_hint and branch_id == latest_hint else " "
            display_summary = summary or "(no summary)"
            print(
                f"  [{index:>2}] {marker} {_short_branch(branch_id)} :: {display_summary}"
            )
        print("  Tip: enter the number, ID, or tag when prompted for branches.")
    else:
        print("\n(No active branches; new replies will create fresh branches.)")
        _LAST_BRANCH_CHOICES = []
    tag_map = tree.get_tag_map()
    if tag_map:
        print("\n🏷️ Known tags:")
        for tag, branch_id in sorted(tag_map.items()):
            print(f"  - {tag} → {_short_branch(branch_id)}")
    print("-" * 60)


def _show_recent_context(manager: DialogueManager) -> None:
    tree = manager.tree
    if len(tree.nodes) <= 1:
        print("\n🧭 Recent focus: (no prior exchanges — ready when you are!)")
        return
    latest = tree.get_latest_active_leaf()
    if not latest or latest == "root":
        print("\n🧭 Recent focus: root context only.")
        return
    node = tree.get_node(latest)
    summary_chain = tree.get_branch_summary_path(latest)
    tags = node.metadata.get("tags") or []
    print("\n🧭 Recent focus:")
    if summary_chain:
        print(f"  Summary chain: {summary_chain}")
    print(f"  Active branch : {_short_branch(latest)} [{node.status}]")
    if tags:
        print(f"  Tags          : {', '.join(tags)}")
    trimmed_query = (node.query or "").strip()
    trimmed_response = (node.response or "").strip()
    if trimmed_query:
        digest = textwrap.shorten(trimmed_query, width=100, placeholder="…")
        print(f"  Last user     : {digest}")
    if trimmed_response:
        digest = textwrap.shorten(trimmed_response, width=100, placeholder="…")
        print(f"  Last AI       : {digest}")


def prompt_manual_decision(
    manager: DialogueManager,
    *,
    thread_tag: Optional[str],
) -> TreeDecision:
    tree = manager.tree
    latest_leaf = tree.get_latest_active_leaf()
    latest_hint = latest_leaf if latest_leaf and latest_leaf != "root" else None

    _describe_active_tree(manager, latest_hint=latest_hint)

    operation = _ask_operation()

    primary: Optional[str] = None
    target_parent: Optional[str] = None
    secondary: List[str] = []
    archive_targets: List[str] = []

    if operation in (TreeOperationType.APPEND, TreeOperationType.MERGE, TreeOperationType.SPLIT):
        need_primary = operation != TreeOperationType.MERGE  # merge can recover later
        primary = _ask_branch(
            f"Primary branch id for {operation.value}",
            default=latest_hint,
            required=need_primary and latest_hint is None,
        )

    if operation == TreeOperationType.MERGE:
        secondary = _ask_branch_list(
            "Secondary branches to merge (comma separated)",
            required=False,
        )
        archive_targets = _ask_branch_list(
            "Branches to archive after merge (comma separated, Enter to skip)",
            required=False,
        )
    elif operation == TreeOperationType.SPLIT:
        target_parent = _ask_branch(
            "Target parent branch (ancestor) for new split (Enter for automatic)",
            default=None,
            required=False,
        )
    elif operation == TreeOperationType.ARCHIVE:
        archive_targets = _ask_branch_list(
            "Branches to archive (comma separated)",
            required=True,
        )

    context_branches = _ask_branch_list(
        "Additional context branches (comma separated, Enter to skip)",
        required=False,
    )
    raw_note = _ask_optional_note()
    header_parts = [f"manual[{operation.value}]"]
    if thread_tag:
        header_parts.append(f"thread={thread_tag}")
    header = " | ".join(header_parts)
    note_body = raw_note.replace("%s", operation.value) if raw_note else "auto"
    note = f"{header} :: {note_body}"

    return TreeDecision(
        operation=operation,
        primary_branch=primary,
        target_parent_branch=target_parent,
        secondary_branches=secondary,
        context_branches=context_branches,
        archive_targets=archive_targets,
        note=note,
    )


class ManualDialogueManager(DialogueManager):
    def llm_cm_decide(self, query: str, thread_tag: Optional[str] = None) -> TreeDecision:
        print("\nManual tree maintenance mode engaged.")
        if thread_tag:
            print(f"Thread tag detected: {thread_tag}")
        return prompt_manual_decision(self, thread_tag=thread_tag)


def build_manual_parser() -> argparse.ArgumentParser:
    parser = build_argument_parser()
    parser.description = (
        "BranchMind Manual Mode — user-driven tree maintenance with LLM responses."
    )
    return parser


def main() -> None:
    parser = build_manual_parser()
    args = parser.parse_args()
    config = AppConfig.from_args(args)
    manager = ManualDialogueManager(config)
    intro = textwrap.dedent(
        """
        BranchMind Manual Mode
        ----------------------
        You are now responsible for choosing how each interaction updates the tree.
        Use branch IDs (full UUID or suffix) or tags (e.g., T-...) when prompted.
        Type 'exit' to quit.
        """
    ).strip()
    print(intro)
    _show_recent_context(manager)

    try:
        while True:
            user_query = input("\nYou: ").strip()
            if user_query.lower() == "exit":
                break
            if not user_query:
                continue
            thread_hint = (
                input("Optional thread tag (e.g., T-123, Enter to skip): ").strip() or None
            )
            try:
                response = manager.handle_request(user_query, thread_id=thread_hint)
                print(f"\n🤖 AI: {response}")
                _show_recent_context(manager)
            except Exception as exc:
                print(f"Error while handling request: {exc}")
    finally:
        manager.save_state()
        print("State saved. Goodbye.")


if __name__ == "__main__":
    main()
