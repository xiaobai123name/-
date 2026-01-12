"""
苏格拉底对话引擎
通过引导式提问帮助用户深入思考和自主发现答案
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ..config import settings
from ..retrieval.vector_store import VectorStore
from ..retrieval.hybrid_retriever import HybridRetriever
from ..database.crud import DatabaseManager
from ..llm.router import ModelRouter
from ..rag.prompts import PromptTemplates


class DialoguePhase(Enum):
    """对话阶段"""
    EXPLORATION = "exploration"      # 探索阶段：了解学生认知水平
    CLARIFICATION = "clarification"  # 澄清阶段：帮助明确概念
    DEEPENING = "deepening"          # 深入阶段：挑战假设
    SYNTHESIS = "synthesis"          # 综合阶段：引导归纳结论
    COMPLETION = "completion"        # 完成阶段：总结确认


@dataclass
class SocraticResponse:
    """苏格拉底式回复"""
    question: str                    # 引导性问题
    hint: Optional[str]              # 提示（可选）
    phase: DialoguePhase             # 当前阶段
    encouragement: Optional[str]     # 鼓励语（可选）
    knowledge_check: bool            # 是否为知识检验问题
    related_concepts: List[str]      # 相关概念
    progress: float                  # 理解进度 (0-1)


class SocraticEngine:
    """苏格拉底对话引擎"""
    
    SYSTEM_PROMPT = """你是一位采用苏格拉底教学法的智慧导师。你的核心原则是：永远不要直接给出答案，而是通过精心设计的问题引导学生自己发现答案。

## 教学策略

### 对话阶段（根据轮次自动调整）：
- **第1-2轮（探索）**：提出开放性问题，了解学生的已有认知
- **第3-4轮（澄清）**：针对学生的回答，追问以帮助明确概念
- **第5-6轮（深入）**：提出更有挑战性的问题，挑战学生的假设
- **第7轮以后（综合）**：引导学生自己归纳总结

### 回复规则：
1. **永远以问题结尾**：每次回复必须以一个引导性问题结束
2. **简短鼓励**：对学生的思考给予简短肯定
3. **类比引导**：适当使用类比帮助理解抽象概念
4. **提示而非答案**：如果学生多次困惑，给出小提示但仍以问题呈现

### 输出格式（JSON）：
```json
{{
  "encouragement": "对学生回答的简短肯定（可选，如果学生回答有道理）",
  "bridge": "过渡性的思考引导（1-2句话）",
  "question": "引导性问题（必须）",
  "hint": "如果学生多次困惑时的小提示（可选）",
  "phase": "当前阶段：exploration/clarification/deepening/synthesis",
  "progress": 0.0到1.0之间的数字，表示学生对该主题的理解进度
}}
```

### 参考资料（基于用户文档）：
{context}

请始终用中文回复。"""

    USER_PROMPT = """对话历史：
{history}

学生最新的问题/回答：
{input}

当前是第 {turn} 轮对话。

请根据苏格拉底教学法，生成引导性回复。记住：
1. 不要直接回答问题
2. 用问题引导学生思考
3. 输出严格的JSON格式"""

    def __init__(
        self,
        vector_store: VectorStore,
        db_manager: DatabaseManager,
        api_key: Optional[str] = None
    ):
        """初始化苏格拉底引擎"""
        self.vector_store = vector_store
        self.db = db_manager
        self.retriever = HybridRetriever(vector_store)
        # 模型路由：按用户+模块动态选择 provider/model
        self.model_router = ModelRouter(db_manager)
    
    def _coerce_content_to_text(self, content: Any) -> str:
        """
        将 LangChain 返回的 message.content 归一化为字符串。

        说明：部分 provider/版本可能返回 list[dict] 的多模态结构（例如 [{"type":"text","text":"..."}]）。
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (bytes, bytearray)):
            try:
                return content.decode("utf-8", errors="ignore")
            except Exception:
                return ""
        if isinstance(content, list):
            parts: List[str] = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                    continue
                if isinstance(p, dict):
                    # 常见结构：{"type":"text","text":"..."}
                    txt = p.get("text") or p.get("content") or ""
                    if isinstance(txt, str) and txt:
                        parts.append(txt)
            return "\n".join([t for t in parts if t]).strip()
        # 兜底：转字符串
        try:
            return str(content)
        except Exception:
            return ""

    def _parse_response(self, response: str) -> Dict:
        """解析LLM的JSON响应"""
        text = self._coerce_content_to_text(response).strip()
        if not text:
            return {"question": "你愿意先说说你目前的理解吗？", "phase": "exploration", "progress": 0.3}

        # 1) 优先提取 code-fence 内的 JSON（兼容 ```json / ```）
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
        json_str = (json_match.group(1) if json_match else text).strip()

        # 兼容：只有开头 ```json 但没有闭合 ``` 的情况
        if not json_match and json_str.lstrip().startswith("```"):
            nl = json_str.find("\n")
            if nl != -1:
                json_str = json_str[nl + 1 :].strip()
            json_str = re.sub(r"\s*```+\s*$", "", json_str).strip()

        # 2) 修复常见 trailing commas
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)

        # 3) 尝试严格 json.loads
        try:
            obj = json.loads(json_str)
            return obj if isinstance(obj, dict) else {"question": text, "phase": "exploration", "progress": 0.3}
        except Exception:
            pass

        # 4) 兜底：raw_decode 解析第一个 JSON 对象，忽略后续噪声
        try:
            # 从第一个 { 开始
            start = json_str.find("{")
            if start >= 0:
                decoder = json.JSONDecoder()
                obj, _end = decoder.raw_decode(json_str[start:])
                if isinstance(obj, dict):
                    return obj
        except Exception:
            pass

        # 5) 最终兜底：把整段文本当作 question
        return {"question": text, "phase": "exploration", "progress": 0.3}
    
    def _format_history(self, messages: List[Dict]) -> str:
        """格式化对话历史"""
        if not messages:
            return "（这是对话的开始）"
        
        history_parts = []
        for msg in messages[-6:]:  # 只保留最近6轮
            role_raw = (msg.get("role") or "").strip().lower()
            role = "学生" if role_raw == "user" else "导师"
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            history_parts.append(f"{role}：{content}")
        
        return "\n".join(history_parts)
    
    def respond(
        self,
        user_input: str,
        user_id: str,
        conversation_history: List[Dict],
        document_ids: Optional[List[str]] = None
    ) -> SocraticResponse:
        """
        生成苏格拉底式回复
        
        Args:
            user_input: 用户输入
            user_id: 用户ID
            conversation_history: 对话历史
            document_ids: 限定文档范围
            
        Returns:
            SocraticResponse: 苏格拉底式回复
        """
        # 计算当前轮次
        turn = len([m for m in (conversation_history or []) if (m.get("role") == "user")]) + 1
        
        # 检索相关文档内容
        chunks = self.retriever.retrieve(
            query=user_input,
            user_id=user_id,
            n_results=5,
            document_ids=document_ids
        )
        
        context = "\n\n".join([
            f"[片段 {i+1}]\n{chunk['content']}" 
            for i, chunk in enumerate(chunks)
            if isinstance(chunk, dict) and (chunk.get("content") or "").strip()
        ]) if chunks else "暂无相关参考资料"
        
        # 格式化对话历史
        history = self._format_history(conversation_history)
        
        # 构建消息
        system_msg = self.SYSTEM_PROMPT.format(context=context)
        user_msg = self.USER_PROMPT.format(
            history=history,
            input=user_input,
            turn=turn
        )
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg)
        ]
        
        # 调用LLM
        llm = self.model_router.get_chat_model(user_id=user_id, module="socratic", streaming=False)
        response = llm.invoke(messages)
        result_raw = self._parse_response(getattr(response, "content", ""))
        result: Dict[str, Any] = result_raw if isinstance(result_raw, dict) else {"question": str(result_raw)}
        
        # 构建回复文本
        reply_parts = []
        if result.get("encouragement"):
            reply_parts.append(result["encouragement"])
        if result.get("bridge"):
            reply_parts.append(result["bridge"])
        
        # 核心问题
        question = result.get("question", "你觉得这个概念的核心是什么？")
        reply_parts.append(f"\n\n**🤔 {question}**")
        
        # 提示（如果有）
        if result.get("hint"):
            reply_parts.append(f"\n\n💡 *小提示：{result['hint']}*")
        
        full_reply = "\n".join(reply_parts)
        
        # 解析阶段
        phase_str = result.get("phase", "exploration")
        try:
            phase = DialoguePhase(phase_str)
        except ValueError:
            phase = DialoguePhase.EXPLORATION

        # progress 归一化
        try:
            progress = float(result.get("progress", 0.3))
        except Exception:
            progress = 0.3
        if progress < 0:
            progress = 0.0
        if progress > 1:
            progress = 1.0
        
        return SocraticResponse(
            question=full_reply,
            hint=result.get("hint"),
            phase=phase,
            encouragement=result.get("encouragement"),
            knowledge_check=turn >= 5,
            related_concepts=[],
            progress=progress
        )
    
    def get_summary(
        self,
        conversation_history: List[Dict],
        user_id: str
    ) -> str:
        """
        生成对话总结
        
        Args:
            conversation_history: 完整对话历史
            user_id: 用户ID
            
        Returns:
            str: 学习总结
        """
        if len(conversation_history) < 4:
            return "对话轮次较少，建议继续深入探讨以获得完整总结。"
        
        history = self._format_history(conversation_history)
        
        summary_prompt = f"""请根据以下苏格拉底式对话，总结学生的学习收获：

{history}

请生成一个简短的学习总结（3-5句话），包括：
1. 学生探索的核心问题
2. 学生通过思考发现的关键点
3. 建议进一步探索的方向

用中文回复。"""
        
        messages = [HumanMessage(content=summary_prompt)]
        llm = self.model_router.get_chat_model(user_id=user_id, module="socratic", streaming=False)
        response = llm.invoke(messages)
        
        return response.content
