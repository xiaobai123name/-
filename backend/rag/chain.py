"""
RAG核心链模块
实现基于LangChain的RAG问答流程
"""

import re
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator, Set
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..config import settings
from ..retrieval.vector_store import VectorStore
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import AdaptiveReranker
from ..database.crud import DatabaseManager
from ..llm.router import ModelRouter
from .prompts import PromptTemplates
from .memory import ConversationMemory


@dataclass
class RAGResponse:
    """RAG响应数据结构"""
    answer: str
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]
    was_reranked: bool
    confidence: float


class RAGChain:
    """RAG核心处理链"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        db_manager: DatabaseManager,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        初始化RAG链
        
        Args:
            vector_store: 向量存储实例
            db_manager: 数据库管理器
            api_key: Google API密钥
            model_name: LLM模型名称
        """
        self.vector_store = vector_store
        self.db = db_manager
        # API key / model_name 将由 ModelRouter 按用户+模块选择，此处保留参数仅作为兼容占位
        self.api_key = api_key
        self.model_name = model_name
        
        # 初始化组件
        self.retriever = HybridRetriever(vector_store)
        self.reranker = AdaptiveReranker()
        self.memory = ConversationMemory(db_manager)

        # 模型路由：按用户+模块动态选择 provider/model
        self.model_router = ModelRouter(db_manager)
    
    def _format_context(
        self,
        chunks: List[Dict[str, Any]],
        use_parent: bool = True
    ) -> str:
        """
        格式化检索到的切片为上下文文本
        
        Args:
            chunks: 检索到的切片列表
            use_parent: 是否使用父切片内容
            
        Returns:
            str: 格式化的上下文
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # 获取内容（优先使用父切片内容，如果有的话）
            content = chunk.get("parent_content") if use_parent and "parent_content" in chunk else chunk["content"]
            
            # 格式化引用信息 - 使用更不易与数学区间混淆的 〔1〕 样式
            source_info = f"〔{i}〕"
            
            context_parts.append(f"{source_info}\n{content}\n")
        
        return "\n---\n".join(context_parts)

    def _build_style_rules(
        self,
        answer_style: str,
        enable_clarify_question: bool = True
    ) -> str:
        """
        构建可选的风格补充规则（用于 UI 侧配置）。

        Args:
            answer_style: teaching / concise / rigorous
            enable_clarify_question: 是否允许在结尾追问一个澄清问题

        Returns:
            str: 追加到 prompt 的规则文本
        """
        style = (answer_style or "").strip().lower()
        rules: List[str] = []

        if style in {"concise", "简洁", "简洁版", "简洁直接"}:
            rules.append("- 简洁优先：少讲铺垫，直奔要点；只保留 1 个最关键例子。")
        elif style in {"rigorous", "exam", "严谨", "考试", "严谨版", "更严谨（偏考试）", "更严谨(偏考试)"}:
            rules.append("- 更严谨：概念边界说清楚；必要时给出定义与反例；避免口语化夸张。")
        else:
            rules.append("- 教学为主：用短句分段，像讲给初中生听；必要时用类比帮助理解。")

        if not enable_clarify_question:
            rules.append("- 不要在结尾提出追问。")

        return "\n".join(rules).strip()

    @staticmethod
    def _strip_code_and_latex(text: str) -> str:
        """
        删除代码块与 LaTeX 片段，降低把数学区间误当成引用角标的概率。
        """
        if not text:
            return ""
        text = re.sub(r"```.*?```", "", text, flags=re.S)
        text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
        text = re.sub(r"\$.*?\$", "", text, flags=re.S)
        return text
    
    def _extract_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        从回答中提取引用信息

        Args:
            answer: LLM生成的回答
            chunks: 检索到的切片列表

        Returns:
            List[Dict]: 引用信息列表，包含文件名、页码、原文片段
        """
        citations = []
        seen_keys = set()  # 用于去重：(document_id, page)

        referenced_indices: Set[int] = set()

        # 1) 优先匹配 〔1〕、〔1,2〕（更不易与数学区间冲突）
        chinese_bracket_pattern = r"〔(\d+(?:,\s*\d+)*)〕"
        chinese_matches = re.findall(chinese_bracket_pattern, answer or "")
        for match in chinese_matches:
            for num_str in match.split(","):
                try:
                    referenced_indices.add(int(num_str.strip()))
                except ValueError:
                    continue

        # 2) 兼容旧格式：[1]、[1,2]（尽量避免把 $P\\in[0,1]$ 这类区间误识别为引用）
        if not referenced_indices:
            scan_text = self._strip_code_and_latex(answer or "")
            square_bracket_pattern = r"\[(\d+(?:,\s*\d+)*)\]"
            square_matches = re.findall(square_bracket_pattern, scan_text)
            for match in square_matches:
                nums = []
                ok = True
                for num_str in match.split(","):
                    try:
                        n = int(num_str.strip())
                        nums.append(n)
                    except ValueError:
                        ok = False
                        break
                # 若包含 0，极大概率是数学区间或编号以外用途，直接跳过整组
                if not ok or (0 in nums):
                    continue
                referenced_indices.update(nums)

        # 如果没有提取到引用，使用所有 chunks（最多取前3个）
        if not referenced_indices:
            referenced_indices = set(range(1, min(len(chunks) + 1, 4)))

        # 为引用的片段获取真正的文件名和内容
        for idx in sorted(referenced_indices):
            if idx < 1 or idx > len(chunks):
                continue

            chunk = chunks[idx - 1]  # 转换为0索引
            metadata = chunk.get("metadata", {})
            document_id = metadata.get("document_id")
            page = metadata.get("start_page")

            # 去重检查
            dedup_key = (document_id, page)
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # 从数据库获取真正的文件名
            filename = "未知文件"
            if document_id:
                doc = self.db.get_document_by_id(document_id)
                if doc:
                    filename = doc.filename

            # 获取原文内容
            content = chunk.get("content", "")

            citations.append({
                "index": idx,  # 片段编号
                "filename": filename,
                "page": page,
                "chunk_id": chunk.get("id"),
                "document_id": document_id,
                "content": content  # 保存原文片段
            })

        return citations
    
    def query(
        self,
        question: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        use_rerank: bool = True,
        top_k: int = 5,
        audience: str = "初中数学",
        answer_style: str = "teaching",
        max_words: int = 350,
        enable_clarify_question: bool = True
    ) -> RAGResponse:
        """
        执行RAG问答
        
        Args:
            question: 用户问题
            user_id: 用户ID
            conversation_id: 对话ID
            document_ids: 限定搜索的文档ID列表
            use_rerank: 是否使用重排序
            top_k: 返回的结果数量
            
        Returns:
            RAGResponse: RAG响应
        """
        # 1. 检索相关文档
        retrieved_chunks = self.retriever.retrieve_with_parent(
            query=question,
            user_id=user_id,
            n_results=top_k * 2,
            document_ids=document_ids
        )
        
        # 2. 自适应重排序
        was_reranked = False
        if use_rerank and retrieved_chunks:
            retrieved_chunks, was_reranked = self.reranker.adaptive_rerank(
                query=question,
                results=retrieved_chunks,
                top_k=top_k
            )
        else:
            retrieved_chunks = retrieved_chunks[:top_k]
        
        # 3. 格式化上下文
        context = self._format_context(retrieved_chunks)
        
        # 4. 获取对话历史
        chat_history = []
        if conversation_id:
            chat_history = self.memory.get_trimmed_history(conversation_id)
        
        # 5. 构建Prompt并调用LLM
        prompt = PromptTemplates.get_rag_prompt()
        
        style_rules = self._build_style_rules(
            answer_style=answer_style,
            enable_clarify_question=enable_clarify_question
        )

        messages = prompt.format_messages(
            context=context,
            question=question,
            chat_history=chat_history,
            audience=audience,
            answer_style=answer_style,
            max_words=max_words,
            style_rules=style_rules
        )
        
        llm = self.model_router.get_chat_model(user_id=user_id, module="rag", streaming=False)
        response = llm.invoke(messages)
        answer = response.content
        
        # 6. 提取引用
        citations = self._extract_citations(answer, retrieved_chunks)
        
        # 7. 计算置信度
        confidence = 0.0
        if retrieved_chunks:
            top_score = retrieved_chunks[0].get("rerank_score") or retrieved_chunks[0].get("fusion_score") or retrieved_chunks[0].get("score", 0)
            confidence = min(top_score, 1.0)
        
        # 8. 保存对话记录
        if conversation_id:
            self.memory.add_message(conversation_id, "user", question)
            self.memory.add_message(conversation_id, "assistant", answer, citations)
        
        return RAGResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            was_reranked=was_reranked,
            confidence=confidence
        )
    
    def stream_query(
        self,
        question: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        use_rerank: bool = True,
        top_k: int = 5,
        audience: str = "初中数学",
        answer_style: str = "teaching",
        max_words: int = 350,
        enable_clarify_question: bool = True
    ) -> Generator[str, None, RAGResponse]:
        """
        流式RAG问答
        
        Args:
            question: 用户问题
            user_id: 用户ID
            conversation_id: 对话ID
            document_ids: 限定搜索的文档ID列表
            use_rerank: 是否使用重排序
            top_k: 返回的结果数量
            
        Yields:
            str: 流式响应的文本片段
            
        Returns:
            RAGResponse: 完整的RAG响应
        """
        # 1. 检索相关文档
        retrieved_chunks = self.retriever.retrieve_with_parent(
            query=question,
            user_id=user_id,
            n_results=top_k * 2,
            document_ids=document_ids
        )
        
        # 2. 自适应重排序
        was_reranked = False
        if use_rerank and retrieved_chunks:
            retrieved_chunks, was_reranked = self.reranker.adaptive_rerank(
                query=question,
                results=retrieved_chunks,
                top_k=top_k
            )
        else:
            retrieved_chunks = retrieved_chunks[:top_k]
        
        # 3. 格式化上下文
        context = self._format_context(retrieved_chunks)
        
        # 4. 获取对话历史
        chat_history = []
        if conversation_id:
            chat_history = self.memory.get_trimmed_history(conversation_id)
        
        # 5. 构建Prompt
        prompt = PromptTemplates.get_rag_prompt()
        style_rules = self._build_style_rules(
            answer_style=answer_style,
            enable_clarify_question=enable_clarify_question
        )

        messages = prompt.format_messages(
            context=context,
            question=question,
            chat_history=chat_history,
            audience=audience,
            answer_style=answer_style,
            max_words=max_words,
            style_rules=style_rules
        )
        
        # 6. 流式调用LLM
        streaming_llm = self.model_router.get_chat_model(user_id=user_id, module="rag", streaming=True)
        full_answer = ""
        for chunk in streaming_llm.stream(messages):
            content = chunk.content
            full_answer += content
            yield content
        
        # 7. 提取引用
        citations = self._extract_citations(full_answer, retrieved_chunks)
        
        # 8. 计算置信度
        confidence = 0.0
        if retrieved_chunks:
            top_score = retrieved_chunks[0].get("rerank_score") or retrieved_chunks[0].get("fusion_score") or retrieved_chunks[0].get("score", 0)
            confidence = min(top_score, 1.0)
        
        # 9. 保存对话记录
        if conversation_id:
            self.memory.add_message(conversation_id, "user", question)
            self.memory.add_message(conversation_id, "assistant", full_answer, citations)
        
        return RAGResponse(
            answer=full_answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            was_reranked=was_reranked,
            confidence=confidence
        )
