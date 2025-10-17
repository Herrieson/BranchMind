import os
import uuid
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# --- 1. 配置 (Configuration) ---
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量

client = AzureOpenAI(
    api_key=os.environ.get("API_KEY"),
    api_version="2024-12-01-preview",  # 注意：版本号可能需要根据您的Azure部署进行调整
    azure_endpoint=os.environ.get("AZURE_ENDPOINT")
)

# 为不同职责选择不同模型
# LLM-CM: 负责决策，需要快速且经济。
CM_MODEL = "gpt-4o-mini"
# LLM-Task: 负责生成最终答案，需要高质量。
TASK_MODEL = "gpt-4o"
# LLM-Summarizer: 用于生成节点摘要，可以和CM模型一样。
SUMMARIZER_MODEL = "gpt-4o-mini"

# --- 2. 数据结构 (Data Structures) ---

@dataclass
class Node:
    """上下文树中的节点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    query: str = ""
    response: str = ""
    summary: str = "" # 对该轮交互的摘要，供CM模型决策使用

class ContextTree:
    """管理上下文树的结构与数据"""
    def __init__(self):
        # 初始化时创建一个根节点
        root_summary = "对话的起点"
        self.root = Node(id="root", summary=root_summary)
        self.nodes: Dict[str, Node] = {"root": self.root}
        self.children_map: Dict[str, List[str]] = {}

    def add_node(self, node: Node):
        """向树中添加一个新节点"""
        if node.parent_id not in self.nodes:
            raise ValueError(f"父节点 {node.parent_id} 不存在。")
        self.nodes[node.id] = node
        # 更新子节点映射表以便快速查找
        if node.parent_id not in self.children_map:
            self.children_map[node.parent_id] = []
        self.children_map[node.parent_id].append(node.id)

    def get_node(self, node_id: str) -> Node:
        """根据ID获取节点"""
        return self.nodes[node_id]

    def get_context_path(self, leaf_id: str) -> List[Dict[str, str]]:
        """从一个叶子节点回溯到根节点，构建上下文路径"""
        if leaf_id not in self.nodes:
            return []
            
        path = []
        curr_node = self.nodes[leaf_id]
        while curr_node.id != "root":
            # 将问答对添加到路径的开头，以保持时间顺序
            path.insert(0, {"role": "assistant", "content": curr_node.response})
            path.insert(0, {"role": "user", "content": curr_node.query})
            if curr_node.parent_id is None: # 理论上不会发生，除非不是从root开始
                break
            curr_node = self.nodes[curr_node.parent_id]
        return path

    def get_active_branches(self) -> Dict[str, str]:
        """获取所有叶子节点（活跃分支）的ID和摘要"""
        # 叶子节点就是那些没有子节点的节点
        leaf_ids = set(self.nodes.keys()) - set(self.children_map.keys())
        return {leaf_id: self.nodes[leaf_id].summary for leaf_id in leaf_ids if leaf_id != "root"}
    
    def get_branch_summary_path(self, leaf_id: str) -> str:
        """从一个叶子节点回溯，构建完整的摘要路径字符串"""
        if leaf_id not in self.nodes:
            return ""
            
        path_summaries = []
        curr_node = self.nodes[leaf_id]
        while curr_node is not None:
            path_summaries.insert(0, curr_node.summary)
            if curr_node.parent_id is None:
                break
            curr_node = self.nodes.get(curr_node.parent_id)
            
        return " -> ".join(path_summaries)

    # ----- 新增功能：打印树状结构 -----
    def display_tree(self):
        """公共方法，用于启动树的打印过程"""
        print("🌳 对话树状结构图:")
        active_ids = set(self.get_active_branches().keys())
        # 从根节点开始递归打印
        self._display_node_recursive(self.root.id, "", True, active_ids)

    def _display_node_recursive(self, node_id: str, prefix: str, is_last: bool, active_ids: set):
        """内部递归方法，用于打印单个节点及其子节点"""
        node = self.get_node(node_id)
        
        # 判断是否为活跃分支
        is_active = node_id in active_ids
        active_marker = " 🍃 [活跃分支]" if is_active else ""
        
        # 准备连接符
        connector = "└── " if is_last else "├── "
        
        # 打印当前节点信息
        print(f"{prefix}{connector}{node.summary} (ID: ...{node_id[-6:]}){active_marker}")
        
        # 获取子节点
        children = self.children_map.get(node_id, [])
        
        # 准备下一层递归的prefix
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # 递归打印子节点
        for i, child_id in enumerate(children):
            is_child_last = (i == len(children) - 1)
            self._display_node_recursive(child_id, new_prefix, is_child_last, active_ids)
    # ------------------------------------

# --- 3. 核心控制器 (DialogueManager) ---

class DialogueManager:
    """负责调度CM和Task模型，管理对话流程"""
    def __init__(self):
        self.tree = ContextTree()
        print("对话管理器已启动。根节点已创建。")

    def _call_llm(self, messages: List[Dict[str, str]], model: str, json_mode: bool = False) -> str:
        """通用的LLM API调用函数"""
        try:
            response_kwargs = {"model": model, "messages": messages}
            if json_mode:
                response_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**response_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"调用API时出错: {e}")
            return "抱歉，处理时遇到错误。"

    def llm_cm_decide(self, query: str) -> Dict:
        """
        调用上下文管理模型(LLM-CM)来决定新查询属于哪个分支。
        """
        active_branches = self.tree.get_active_branches()
        
        # 如果没有活跃分支（只有根节点），强制创建新分支
        if not active_branches:
            return {"decision": "new", "branch_id": self.tree.root.id}
        
        # 为每个活跃分支生成完整的摘要路径
        branch_full_paths = {
            branch_id: self.tree.get_branch_summary_path(branch_id)
            for branch_id in active_branches.keys()
        }

        # 构建给CM模型的提示
        prompt = f"""
        你是一个专业的对话上下文管理器。你的任务是根据用户的最新提问，决定它应该属于哪个已存在的对话分支，还是应该创建一个全新的分支。

        用户的最新提问是: "{query}"

        这里是当前所有活跃的对话分支的完整摘要路径（key是分支ID，value是摘要路径）：
        {json.dumps(branch_full_paths, indent=2, ensure_ascii=False)}

        请仔细分析用户的提问与上述哪个摘要路径在逻辑上最为相关。
        1.  如果提问是某个分支的延续（例如，对之前回答的追问、补充、修正），请选择 'append'。
        2.  如果提问开启了一个完全不相关的新主题，或者用户明确表示要开始新的探索，请选择 'new'。

        请以严格的JSON格式返回你的决定，格式如下：
        {{
          "decision": "append" | "new",
          "branch_id": "如果决定是append，请填上你选择的branch_id；如果是new，请填null"
        }}
        """
        
        messages = [{"role": "system", "content": "你是一个帮助分类用户请求的AI助手，总是以JSON格式输出。"},
                    {"role": "user", "content": prompt}]
        
        decision_str = self._call_llm(messages, model=CM_MODEL, json_mode=True)
        
        try:
            decision = json.loads(decision_str)
            # 简单的验证
            if decision.get("decision") == "append" and decision.get("branch_id") not in active_branches:
                 # 如果模型给出了一个无效的分支ID，则强制新建分支
                print(f"警告: CM模型选择了无效的分支ID '{decision.get('branch_id')}'。强制创建新分支。")
                decision = {"decision": "new", "branch_id": None}
            return decision
        except (json.JSONDecodeError, TypeError):
            print(f"错误: CM模型返回的不是有效的JSON: {decision_str}")
            # 出错时默认创建新分支
            return {"decision": "new", "branch_id": self.tree.root.id}

    def llm_task_execute(self, query: str, context: List[Dict[str, str]]) -> str:
        """
        调用任务生成模型(LLM-Task)来生成最终的回复。
        """
        system_message = "你是一个能干的AI助手，请根据下面提供的上下文历史和用户的最新问题，给出清晰、准确的回答。"
        messages = [{"role": "system", "content": system_message}] + context + [{"role": "user", "content": query}]
        
        response = self._call_llm(messages, model=TASK_MODEL)
        return response

    def _summarize_interaction(self, query: str, response: str) -> str:
        """为一轮新的问答生成摘要"""
        prompt = f"请为以下这轮对话生成一个简洁的、不超过20个字的摘要，用于后续的逻辑判断。\n\n用户问：{query}\nAI答：{response}"
        messages = [{"role": "system", "content": "你是一个文本摘要专家。"},
                    {"role": "user", "content": prompt}]
        summary = self._call_llm(messages, model=SUMMARIZER_MODEL)
        return summary.strip().replace("\n", " ")

    def handle_request(self, query: str):
        """处理一个用户请求的完整流程"""
        print("\n" + "="*50)
        print(f"接收到新请求: {query}")

        # 1. 分支决策
        decision = self.llm_cm_decide(query)
        print(f"🧠 LLM-CM 决策: {decision}")

        # 2. 确定父节点
        parent_id = None
        if decision.get("decision") == "new":
            parent_id = self.tree.root.id
            print("   └─ 决策结果：创建新分支，父节点为 'root'")
        else:
            parent_id = decision.get("branch_id")
            print(f"   └─ 决策结果：附加到现有分支 '...{parent_id[-6:]}'")

        if not parent_id:
             print("警告：决策后未能确定父节点，强制使用root作为父节点。")
             parent_id = self.tree.root.id

        # 3. 上下文生成
        context_path = self.tree.get_context_path(parent_id)
        
        # 4. 任务执行
        print("🚀 正在调用 LLM-Task 生成回复...")
        response = self.llm_task_execute(query, context_path)
        
        # 5. 生成摘要并更新树
        print("📝 正在生成交互摘要...")
        summary = self._summarize_interaction(query, response)
        new_node = Node(parent_id=parent_id, query=query, response=response, summary=summary)
        self.tree.add_node(new_node)
        print(f"   └─ 新节点 '...{new_node.id[-6:]}' 已创建，摘要为: '{summary}'")
        
        # 6. (新增) 打印当前的树状结构
        self.tree.display_tree()
        
        print("="*50 + "\n")
        return response

# --- 4. 运行示例 (main) ---
if __name__ == "__main__":
    manager = DialogueManager()

    print("\n欢迎来到动态树状上下文管理系统 (V2)！")
    print("系统现在会在每次交互后实时打印对话树的结构图。")
    print("您可以尝试以下类型的对话来测试系统：")
    print(" - 开启一个话题, 如: '你好，请帮我规划一个为期三天的东京旅游行程。'")
    print(" - 在该话题上追问, 如: '第一天有什么美食推荐吗？'")
    print(" - 开启一个完全不相关的新话题, 如: '能给我写一个Python的快速排序算法吗？'")
    print(" - 回到之前的话题, 如: '关于之前那个东京行程，第二天住在哪里比较方便？'")
    print(" - 输入 'exit' 退出程序。")
    print("-" * 20)
    
    while True:
        user_query = input("你: ")
        if user_query.lower() == 'exit':
            break
        
        ai_response = manager.handle_request(user_query)
        print(f"AI: {ai_response}")