import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from branchmind.main import AppConfig, DialogueManager, Node, build_argument_parser


STATIC_DIR = Path(__file__).resolve().parent / "static"


class CreateNodePayload(BaseModel):
    parent_id: Optional[str] = Field(default="root")
    query: str = Field(..., min_length=1, description="User question stored on the node.")


class MoveNodePayload(BaseModel):
    new_parent_id: str = Field(..., description="Target parent branch ID.")


class ChatNodeResponse(BaseModel):
    node: Dict[str, Any]


def _build_manager() -> DialogueManager:
    parser = build_argument_parser()
    args = parser.parse_args([])
    config = AppConfig.from_args(args)
    return DialogueManager(config)


MANAGER = _build_manager()
TREE_LOCK = asyncio.Lock()


def _serialize_node(node: Node) -> Dict[str, Any]:
    return {
        "id": node.id,
        "parent_id": node.parent_id,
        "query": node.query,
        "response": node.response,
        "summary": node.summary,
        "status": node.status,
        "metadata": node.metadata,
    }


app = FastAPI(title="BranchMind Tree Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI assets not found.")
    return FileResponse(index_path)


@app.get("/api/tree")
async def get_tree() -> Dict[str, Any]:
    async with TREE_LOCK:
        tree = MANAGER.tree
        nodes = {node_id: _serialize_node(node) for node_id, node in tree.nodes.items()}
        return {
            "root_id": tree.root.id,
            "nodes": nodes,
            "children_map": tree.children_map,
        }


@app.get("/api/nodes/{node_id}")
async def get_node(node_id: str) -> Dict[str, Any]:
    async with TREE_LOCK:
        if node_id not in MANAGER.tree.nodes:
            raise HTTPException(status_code=404, detail="Node not found.")
        node = MANAGER.tree.nodes[node_id]
        return {"node": _serialize_node(node)}


@app.post("/api/nodes", response_model=ChatNodeResponse)
async def create_node(payload: CreateNodePayload) -> ChatNodeResponse:
    async with TREE_LOCK:
        parent_id = payload.parent_id or "root"
        if parent_id not in MANAGER.tree.nodes:
            raise HTTPException(status_code=400, detail="Parent branch does not exist.")
        node = Node(
            parent_id=parent_id,
            query=payload.query.strip(),
            response="",
            summary="",
            metadata={"created_by": "user"},
        )
        MANAGER.tree.add_node(node)
        MANAGER.save_state()
        return ChatNodeResponse(node=_serialize_node(node))


@app.post("/api/nodes/{node_id}/move", response_model=ChatNodeResponse)
async def move_node(node_id: str, payload: MoveNodePayload) -> ChatNodeResponse:
    async with TREE_LOCK:
        tree = MANAGER.tree
        if node_id == "root":
            raise HTTPException(status_code=400, detail="Root node cannot be moved.")
        if node_id not in tree.nodes:
            raise HTTPException(status_code=404, detail="Node not found.")
        target_parent = payload.new_parent_id
        if target_parent not in tree.nodes:
            raise HTTPException(status_code=400, detail="Target parent not found.")
        # Prevent creating cycles.
        if tree.is_ancestor(node_id, target_parent):
            raise HTTPException(status_code=400, detail="Cannot move node under its descendant.")
        node = tree.nodes[node_id]
        old_parent = node.parent_id
        if old_parent and old_parent in tree.children_map:
            tree.children_map[old_parent] = [cid for cid in tree.children_map[old_parent] if cid != node_id]
        node.parent_id = target_parent
        tree.children_map.setdefault(target_parent, [])
        if node_id not in tree.children_map[target_parent]:
            tree.children_map[target_parent].append(node_id)
        MANAGER.save_state()
        return ChatNodeResponse(node=_serialize_node(node))


@app.post("/api/nodes/{node_id}/chat", response_model=ChatNodeResponse)
async def chat_with_node(node_id: str) -> ChatNodeResponse:
    async with TREE_LOCK:
        tree = MANAGER.tree
        if node_id not in tree.nodes:
            raise HTTPException(status_code=404, detail="Node not found.")
        node = tree.nodes[node_id]
        if not node.query.strip():
            raise HTTPException(status_code=400, detail="Node must contain a user query before chatting.")

        context_messages = tree.get_context_path(node_id)
        # remove trailing empty assistant message if node has no response yet
        if context_messages and context_messages[-1]["role"] == "assistant" and not node.response:
            context_messages = context_messages[:-1]
        response_text = MANAGER.llm_task_execute(node.query, context_messages)
        summary = MANAGER._summarize_interaction(node.query, response_text)
        node.response = response_text
        node.summary = summary
        MANAGER.save_state()
        return ChatNodeResponse(node=_serialize_node(node))


@app.post("/api/nodes/{node_id}/archive", response_model=ChatNodeResponse)
async def archive_node(node_id: str) -> ChatNodeResponse:
    async with TREE_LOCK:
        tree = MANAGER.tree
        if node_id not in tree.nodes:
            raise HTTPException(status_code=404, detail="Node not found.")
        tree.archive_branch(node_id)
        MANAGER.save_state()
        node = tree.nodes[node_id]
        return ChatNodeResponse(node=_serialize_node(node))


@app.delete("/api/nodes/{node_id}")
async def delete_node(node_id: str) -> Dict[str, Any]:
    async with TREE_LOCK:
        tree = MANAGER.tree
        if node_id == "root":
            raise HTTPException(status_code=400, detail="Cannot delete root.")
        if node_id not in tree.nodes:
            raise HTTPException(status_code=404, detail="Node not found.")
        # Only allow deleting leaves to avoid complex cascade logic.
        children = tree.children_map.get(node_id, [])
        if children:
            raise HTTPException(status_code=400, detail="Only leaf nodes can be deleted.")
        node = tree.nodes[node_id]
        parent_id = node.parent_id
        if parent_id and parent_id in tree.children_map:
            tree.children_map[parent_id] = [cid for cid in tree.children_map[parent_id] if cid != node_id]
        del tree.nodes[node_id]
        MANAGER.save_state()
        return {"status": "ok", "deleted": node_id}
