const state = {
  tree: null,
  selectedNodeId: null,
  draftParentId: null,
};

const treeContainer = document.getElementById("tree-container");
const detailPanel = document.getElementById("node-details");
const statusIndicator = document.getElementById("status-indicator");
const questionInput = document.getElementById("question-input");
const addNodeBtn = document.getElementById("add-node-btn");
const chatBtn = document.getElementById("chat-btn");

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail.detail || response.statusText;
    throw new Error(message);
  }
  return response.json();
}

function setStatus(text) {
  statusIndicator.textContent = text;
}

function renderTree() {
  treeContainer.innerHTML = "";
  if (!state.tree) return;

  const { root_id: rootId, nodes, children_map: childrenMap } = state.tree;

  function buildBranch(nodeId) {
    const node = nodes[nodeId];
    const wrapper = document.createElement("div");
    wrapper.className = "tree-node";

    const button = document.createElement("button");
    button.textContent =
      (node.summary && node.summary.trim()) ||
      (node.query && node.query.trim()) ||
      "(空节点)";
    button.dataset.nodeId = nodeId;
    button.draggable = nodeId !== "root";

    const meta = document.createElement("span");
    meta.className = "node-meta";
    meta.textContent = `${nodeId.slice(-6)} • ${node.status}`;

    button.addEventListener("click", () => {
      selectNode(nodeId);
    });

    button.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("application/node-id", nodeId);
      event.dataTransfer.effectAllowed = "move";
    });

    button.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
    });

    button.addEventListener("drop", async (event) => {
      event.preventDefault();
      const sourceId = event.dataTransfer.getData("application/node-id");
      if (!sourceId || sourceId === nodeId) {
        return;
      }
      try {
        setStatus("移动节点中...");
        await fetchJSON(`/api/nodes/${encodeURIComponent(sourceId)}/move`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ new_parent_id: nodeId }),
        });
        await loadTree(nodeId);
        setStatus("节点已移动");
      } catch (error) {
        console.error(error);
        alert(`移动失败: ${error.message}`);
        setStatus("操作失败");
      }
    });

    if (state.selectedNodeId === nodeId) {
      button.classList.add("active");
    }

    wrapper.appendChild(button);
    wrapper.appendChild(meta);

    const children = childrenMap[nodeId] || [];
    if (children.length > 0) {
      const list = document.createElement("div");
      children.forEach((childId) => {
        list.appendChild(buildBranch(childId));
      });
      wrapper.appendChild(list);
    }

    return wrapper;
  }

  treeContainer.appendChild(buildBranch(rootId));
}

function renderNodeDetails(nodeId) {
  if (!state.tree) return;
  const node = state.tree.nodes[nodeId];
  if (!node) return;

  detailPanel.innerHTML = "";
  const heading = document.createElement("h2");
  heading.textContent = node.summary || "(未生成摘要)";

  const toolbar = document.createElement("div");
  toolbar.className = "toolbar";

  const archiveBtn = document.createElement("button");
  archiveBtn.textContent = "归档节点";
  archiveBtn.addEventListener("click", async () => {
    try {
      setStatus("归档中...");
      await fetchJSON(`/api/nodes/${encodeURIComponent(nodeId)}/archive`, {
        method: "POST",
      });
      await loadTree(nodeId);
      setStatus("节点已归档");
    } catch (error) {
      alert(`归档失败: ${error.message}`);
      setStatus("操作失败");
    }
  });

  const deleteBtn = document.createElement("button");
  deleteBtn.textContent = "删除节点";
  deleteBtn.addEventListener("click", async () => {
    if (!confirm("确定删除该节点？仅支持删除叶子节点。")) return;
    try {
      setStatus("删除中...");
      await fetchJSON(`/api/nodes/${encodeURIComponent(nodeId)}`, {
        method: "DELETE",
      });
      state.selectedNodeId = node.parent_id || "root";
      await loadTree(state.selectedNodeId);
      setStatus("节点已删除");
    } catch (error) {
      alert(`删除失败: ${error.message}`);
      setStatus("操作失败");
    }
  });

  toolbar.appendChild(archiveBtn);
  toolbar.appendChild(deleteBtn);

  const fields = [
    ["节点 ID", node.id],
    ["父节点", node.parent_id || "(无)"],
    ["状态", node.status],
    ["用户提问", node.query || "(未填写)"],
    ["模型回答", node.response || "(尚未调用模型)"],
    ["摘要", node.summary || "(尚未生成)"],
    [
      "标签",
      Array.isArray(node.metadata?.tags)
        ? node.metadata.tags.join(", ")
        : "(无标签)",
    ],
    ["原始元数据", JSON.stringify(node.metadata || {}, null, 2)],
  ];

  detailPanel.appendChild(heading);
  detailPanel.appendChild(toolbar);

  fields.forEach(([label, value]) => {
    const wrapper = document.createElement("div");
    wrapper.className = "detail-field";
    const labelElem = document.createElement("label");
    labelElem.textContent = label;
    const pre = document.createElement("pre");
    pre.textContent = value;
    wrapper.appendChild(labelElem);
    wrapper.appendChild(pre);
    detailPanel.appendChild(wrapper);
  });
}

async function selectNode(nodeId) {
  state.selectedNodeId = nodeId;
  renderTree();
  renderNodeDetails(nodeId);
}

async function loadTree(preferredSelection) {
  try {
    setStatus("加载树数据...");
    const data = await fetchJSON("/api/tree");
    state.tree = data;
    renderTree();
    setStatus("就绪");
    if (preferredSelection) {
      selectNode(preferredSelection);
    } else if (state.selectedNodeId) {
      selectNode(state.selectedNodeId);
    }
  } catch (error) {
    console.error(error);
    setStatus(`加载失败: ${error.message}`);
  }
}

addNodeBtn.addEventListener("click", async () => {
  const query = questionInput.value.trim();
  if (!query) {
    alert("请输入新的用户提问。");
    return;
  }
  const parentId = state.selectedNodeId || "root";
  try {
    setStatus("新增节点...");
    const result = await fetchJSON("/api/nodes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parent_id: parentId, query }),
    });
    questionInput.value = "";
    await loadTree(result.node.id);
    setStatus("节点已创建");
  } catch (error) {
    alert(`创建失败: ${error.message}`);
    setStatus("操作失败");
  }
});

chatBtn.addEventListener("click", async () => {
  const nodeId = state.selectedNodeId;
  if (!nodeId) {
    alert("请先选择一个节点。");
    return;
  }
  const node = state.tree.nodes[nodeId];
  if (!node || !node.query.trim()) {
    alert("该节点尚未填写用户提问。");
    return;
  }
  try {
    setStatus("向模型提问...");
    await fetchJSON(`/api/nodes/${encodeURIComponent(nodeId)}/chat`, {
      method: "POST",
    });
    await loadTree(nodeId);
    setStatus("模型回复已更新");
  } catch (error) {
    alert(`调用模型失败: ${error.message}`);
    setStatus("操作失败");
  }
});

loadTree("root");

