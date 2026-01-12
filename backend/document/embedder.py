"""
文档向量化模块
使用Google text-embedding-004进行文本嵌入
"""

import asyncio
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

from ..config import settings


class DocumentEmbedder:
    """文档向量化器"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: str = "models/text-embedding-004"
    ):
        """
        初始化向量化器
        
        Args:
            api_key: Google API密钥
            model_name: 嵌入模型名称
        """
        self.api_key = api_key or settings.GOOGLE_API_KEY
        self.model_name = model_name
        
        # 配置Google API
        if self.api_key:
            genai.configure(api_key=self.api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_text(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        将单个文本转换为向量
        
        Args:
            text: 输入文本
            task_type: 任务类型
                - "retrieval_document": 用于文档存储
                - "retrieval_query": 用于查询
                - "semantic_similarity": 语义相似度
                - "classification": 分类任务
                
        Returns:
            List[float]: 嵌入向量
        """
        if not text or not text.strip():
            raise ValueError("文本不能为空")
        
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=task_type
        )
        
        return result['embedding']
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def embed_texts(
        self, 
        texts: List[str], 
        task_type: str = "retrieval_document",
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        批量将文本转换为向量
        
        Args:
            texts: 文本列表
            task_type: 任务类型
            batch_size: 每批处理的文本数量
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not texts:
            return []
        
        # 过滤空文本
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            return []
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            
            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type=task_type
            )
            
            all_embeddings.extend(result['embedding'])
        
        return all_embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        将查询文本转换为向量（用于检索）
        
        Args:
            query: 查询文本
            
        Returns:
            List[float]: 查询嵌入向量
        """
        return self.embed_text(query, task_type="retrieval_query")
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        将文档列表转换为向量（用于存储）
        
        Args:
            documents: 文档文本列表
            
        Returns:
            List[List[float]]: 文档嵌入向量列表
        """
        return self.embed_texts(documents, task_type="retrieval_document")
    
    @property
    def embedding_dimension(self) -> int:
        """获取嵌入向量维度"""
        # text-embedding-004 的维度是 768
        return 768
