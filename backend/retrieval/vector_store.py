"""
向量存储模块
基于ChromaDB实现向量存储和检索
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings

from ..config import settings
from ..document.embedder import DocumentEmbedder
from ..document.chunker import ChunkData


class VectorStore:
    """ChromaDB向量存储封装"""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "documents",
        embedder: Optional[DocumentEmbedder] = None
    ):
        """
        初始化向量存储
        
        Args:
            persist_directory: 持久化目录
            collection_name: 集合名称
            embedder: 文档向量化器
        """
        self.persist_directory = persist_directory or str(settings.chroma_path)
        self.collection_name = collection_name
        self.embedder = embedder or DocumentEmbedder()
        
        # 确保目录存在
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
    
    def add_chunks(
        self,
        chunks: List[ChunkData],
        user_id: str,
        show_progress: bool = False
    ) -> int:
        """
        添加文档切片到向量存储
        
        Args:
            chunks: 切片列表
            user_id: 用户ID
            show_progress: 是否显示进度
            
        Returns:
            int: 成功添加的切片数量
        """
        if not chunks:
            return 0
        
        # 准备数据
        ids = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            ids.append(chunk.id)
            documents.append(chunk.content)
            metadatas.append({
                "user_id": user_id,
                "document_id": chunk.document_id,
                "chunk_type": chunk.chunk_type,
                "parent_chunk_id": chunk.parent_chunk_id or "",
                "chunk_index": chunk.chunk_index,
                "start_page": chunk.start_page or 0,
                "end_page": chunk.end_page or 0,
                "section_title": chunk.section_title or "",
                "token_count": chunk.token_count or 0
            })
        
        # 生成嵌入向量
        embeddings = self.embedder.embed_documents(documents)
        
        # 批量添加到ChromaDB
        batch_size = 100
        added_count = 0
        
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            added_count += end_idx - i
        
        return added_count
    
    def search(
        self,
        query: str,
        user_id: str,
        n_results: int = 10,
        chunk_type: Optional[str] = "child",
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        
        Args:
            query: 查询文本
            user_id: 用户ID
            n_results: 返回结果数量
            chunk_type: 切片类型筛选 ("child", "parent", None)
            document_ids: 限定搜索的文档ID列表
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        # 生成查询向量
        query_embedding = self.embedder.embed_query(query)
        
        # 构建过滤条件
        where_conditions = {"user_id": user_id}
        
        if chunk_type:
            where_conditions["chunk_type"] = chunk_type
        
        # ChromaDB的where语法
        where = {"$and": [{"user_id": {"$eq": user_id}}]}
        
        if chunk_type:
            where["$and"].append({"chunk_type": {"$eq": chunk_type}})
        
        if document_ids:
            where["$and"].append({"document_id": {"$in": document_ids}})
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where if len(where["$and"]) > 1 else {"user_id": {"$eq": user_id}},
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": chunk_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # 转换为相似度分数
                    "distance": results["distances"][0][i]
                })
        
        return formatted_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """根据ID获取切片"""
        results = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if results and results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None
    
    def get_parent_chunk(self, child_chunk_id: str) -> Optional[Dict[str, Any]]:
        """获取子切片对应的父切片"""
        child = self.get_chunk_by_id(child_chunk_id)
        if child and child["metadata"].get("parent_chunk_id"):
            return self.get_chunk_by_id(child["metadata"]["parent_chunk_id"])
        return None
    
    def delete_document_chunks(self, document_id: str, user_id: str) -> int:
        """
        删除指定文档的所有切片
        
        Args:
            document_id: 文档ID
            user_id: 用户ID
            
        Returns:
            int: 删除的切片数量
        """
        # 查找所有相关切片
        results = self.collection.get(
            where={
                "$and": [
                    {"user_id": {"$eq": user_id}},
                    {"document_id": {"$eq": document_id}}
                ]
            }
        )
        
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        
        return 0
    
    def delete_user_data(self, user_id: str) -> int:
        """
        删除用户的所有数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            int: 删除的切片数量
        """
        results = self.collection.get(
            where={"user_id": {"$eq": user_id}}
        )
        
        if results and results["ids"]:
            self.collection.delete(ids=results["ids"])
            return len(results["ids"])
        
        return 0
    
    def get_collection_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """获取集合统计信息"""
        if user_id:
            results = self.collection.get(
                where={"user_id": {"$eq": user_id}}
            )
            count = len(results["ids"]) if results and results["ids"] else 0
        else:
            count = self.collection.count()
        
        return {
            "total_chunks": count,
            "collection_name": self.collection_name
        }
