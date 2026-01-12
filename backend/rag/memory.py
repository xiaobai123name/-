"""
对话记忆管理模块
实现对话历史的存储、检索和智能摘要
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

from ..database.crud import DatabaseManager


class ConversationMemory:
    """对话记忆管理器"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        max_history_tokens: int = 2000,
        summary_threshold: int = 10
    ):
        """
        初始化对话记忆管理器
        
        Args:
            db_manager: 数据库管理器
            max_history_tokens: 最大历史Token数
            summary_threshold: 触发摘要的消息数阈值
        """
        self.db = db_manager
        self.max_history_tokens = max_history_tokens
        self.summary_threshold = summary_threshold
        
        # 内存中的短期记忆 {conversation_id: [messages]}
        self._short_term_memory: Dict[str, List[Dict]] = {}
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict]] = None
    ) -> None:
        """
        添加消息到对话记忆
        
        Args:
            conversation_id: 对话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            citations: 引用信息
        """
        # 保存到数据库
        self.db.add_message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations
        )
        
        # 更新短期记忆
        if conversation_id not in self._short_term_memory:
            self._short_term_memory[conversation_id] = []
        
        self._short_term_memory[conversation_id].append({
            "role": role,
            "content": content,
            "citations": citations,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_chat_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        获取对话历史（LangChain格式）
        
        Args:
            conversation_id: 对话ID
            limit: 限制返回的消息数量
            
        Returns:
            List[BaseMessage]: LangChain消息列表
        """
        # 优先从短期记忆获取
        if conversation_id in self._short_term_memory:
            messages = self._short_term_memory[conversation_id]
        else:
            # 从数据库加载
            db_messages = self.db.get_conversation_messages(conversation_id, limit)
            messages = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "citations": msg.citations
                }
                for msg in db_messages
            ]
            self._short_term_memory[conversation_id] = messages
        
        if limit:
            messages = messages[-limit:]
        
        # 转换为LangChain格式
        chat_history = []
        for msg in messages:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                chat_history.append(SystemMessage(content=msg["content"]))
        
        return chat_history
    
    def get_recent_context(
        self,
        conversation_id: str,
        max_messages: int = 6
    ) -> str:
        """
        获取最近的对话上下文（文本格式）
        
        Args:
            conversation_id: 对话ID
            max_messages: 最大消息数
            
        Returns:
            str: 格式化的对话上下文
        """
        messages = self.get_chat_history(conversation_id, limit=max_messages)
        
        context_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"用户：{msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"助手：{msg.content}")
        
        return "\n".join(context_parts)
    
    def clear_conversation(self, conversation_id: str) -> None:
        """
        清除对话记忆
        
        Args:
            conversation_id: 对话ID
        """
        if conversation_id in self._short_term_memory:
            del self._short_term_memory[conversation_id]
    
    def get_conversation_summary(self, conversation_id: str) -> Optional[str]:
        """
        获取对话摘要
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            Optional[str]: 对话摘要
        """
        # 从短期记忆获取消息
        if conversation_id in self._short_term_memory:
            messages = self._short_term_memory[conversation_id]
        else:
            return None
        
        if len(messages) < 3:
            return None
        
        # 简单的摘要：提取关键问题
        user_messages = [m["content"] for m in messages if m["role"] == "user"]
        if user_messages:
            # 取最近的几个问题作为摘要
            recent_questions = user_messages[-3:]
            return "讨论主题：" + "；".join([q[:50] + "..." if len(q) > 50 else q for q in recent_questions])
        
        return None
    
    def estimate_tokens(self, text: str) -> int:
        """估算文本的Token数"""
        # 简单估算：中文约1.5字符/token，英文约4字符/token
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)
    
    def get_trimmed_history(
        self,
        conversation_id: str,
        max_tokens: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        获取修剪后的对话历史（控制Token数量）
        
        Args:
            conversation_id: 对话ID
            max_tokens: 最大Token数
            
        Returns:
            List[BaseMessage]: 修剪后的消息列表
        """
        max_tokens = max_tokens or self.max_history_tokens
        all_messages = self.get_chat_history(conversation_id)
        
        if not all_messages:
            return []
        
        # 从最近的消息开始，累计Token数
        trimmed = []
        total_tokens = 0
        
        for msg in reversed(all_messages):
            msg_tokens = self.estimate_tokens(msg.content)
            if total_tokens + msg_tokens <= max_tokens:
                trimmed.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
        
        return trimmed
