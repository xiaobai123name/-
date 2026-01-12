"""
混合检索器
结合向量检索和BM25关键词检索，使用RRF进行结果融合
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from rank_bm25 import BM25Okapi
import jieba

from .vector_store import VectorStore


class HybridRetriever:
    """混合检索器：向量检索 + BM25 + RRF融合"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            rrf_k: RRF融合参数k
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        
        # BM25索引缓存 {user_id: {document_ids: (bm25, corpus, chunk_ids)}}
        self._bm25_cache: Dict[str, Dict[str, Tuple[BM25Okapi, List[List[str]], List[str]]]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """
        文本分词（支持中英文混合）
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        # 使用jieba进行中文分词
        tokens = list(jieba.cut(text))
        
        # 过滤停用词和标点
        tokens = [
            t.strip().lower() 
            for t in tokens 
            if t.strip() and len(t.strip()) > 1 and not re.match(r'^[\W]+$', t)
        ]
        
        return tokens
    
    def _build_bm25_index(
        self,
        user_id: str,
        document_ids: Optional[List[str]] = None
    ) -> Tuple[Optional[BM25Okapi], List[List[str]], List[str], List[Dict]]:
        """
        构建BM25索引
        
        Args:
            user_id: 用户ID
            document_ids: 文档ID列表
            
        Returns:
            Tuple: (BM25索引, 分词语料库, 切片ID列表, 切片数据列表)
        """
        # 从ChromaDB获取所有相关切片
        where_conditions = {"user_id": {"$eq": user_id}}
        
        if document_ids:
            # 需要分批获取，因为ChromaDB可能有数量限制
            all_results = {"ids": [], "documents": [], "metadatas": []}
            
            for doc_id in document_ids:
                results = self.vector_store.collection.get(
                    where={
                        "$and": [
                            {"user_id": {"$eq": user_id}},
                            {"document_id": {"$eq": doc_id}},
                            {"chunk_type": {"$eq": "child"}}
                        ]
                    },
                    include=["documents", "metadatas"]
                )
                if results and results["ids"]:
                    all_results["ids"].extend(results["ids"])
                    all_results["documents"].extend(results["documents"])
                    all_results["metadatas"].extend(results["metadatas"])
            
            results = all_results
        else:
            results = self.vector_store.collection.get(
                where={
                    "$and": [
                        {"user_id": {"$eq": user_id}},
                        {"chunk_type": {"$eq": "child"}}
                    ]
                },
                include=["documents", "metadatas"]
            )
        
        if not results or not results["ids"]:
            return None, [], [], []
        
        # 构建语料库
        corpus = []
        chunk_ids = results["ids"]
        chunk_data = []
        
        for i, doc in enumerate(results["documents"]):
            tokens = self._tokenize(doc)
            corpus.append(tokens)
            chunk_data.append({
                "id": results["ids"][i],
                "content": doc,
                "metadata": results["metadatas"][i]
            })
        
        # 创建BM25索引
        if corpus:
            bm25 = BM25Okapi(corpus)
            return bm25, corpus, chunk_ids, chunk_data
        
        return None, [], [], []
    
    def _rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Tuple[str, float, Dict]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        使用Reciprocal Rank Fusion (RRF) 融合多路检索结果
        
        Args:
            vector_results: 向量检索结果
            bm25_results: BM25检索结果 [(chunk_id, score, data), ...]
            k: RRF参数
            
        Returns:
            List[Dict]: 融合后的排序结果
        """
        rrf_scores = defaultdict(float)
        chunk_data = {}
        
        # 处理向量检索结果
        for rank, result in enumerate(vector_results):
            chunk_id = result["id"]
            rrf_scores[chunk_id] += self.vector_weight * (1.0 / (k + rank + 1))
            chunk_data[chunk_id] = result
        
        # 处理BM25结果
        for rank, (chunk_id, bm25_score, data) in enumerate(bm25_results):
            rrf_scores[chunk_id] += self.bm25_weight * (1.0 / (k + rank + 1))
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = {
                    "id": chunk_id,
                    "content": data["content"],
                    "metadata": data["metadata"],
                    "score": 0,
                    "bm25_score": bm25_score
                }
            else:
                chunk_data[chunk_id]["bm25_score"] = bm25_score
        
        # 按RRF分数排序
        sorted_results = []
        for chunk_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: -x[1]):
            result = chunk_data[chunk_id].copy()
            result["rrf_score"] = rrf_score
            result["fusion_score"] = rrf_score
            sorted_results.append(result)
        
        return sorted_results
    
    def retrieve(
        self,
        query: str,
        user_id: str,
        n_results: int = 10,
        document_ids: Optional[List[str]] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索
        
        Args:
            query: 查询文本
            user_id: 用户ID
            n_results: 返回结果数量
            document_ids: 限定搜索的文档ID列表
            use_hybrid: 是否使用混合检索（False则只用向量检索）
            
        Returns:
            List[Dict]: 检索结果列表
        """
        # 1. 向量检索
        vector_results = self.vector_store.search(
            query=query,
            user_id=user_id,
            n_results=n_results * 2,  # 多检索一些用于融合
            chunk_type="child",
            document_ids=document_ids
        )
        
        if not use_hybrid:
            return vector_results[:n_results]
        
        # 2. BM25检索
        bm25, corpus, chunk_ids, chunk_data = self._build_bm25_index(user_id, document_ids)
        
        bm25_results = []
        if bm25 and corpus:
            query_tokens = self._tokenize(query)
            scores = bm25.get_scores(query_tokens)
            
            # 获取top结果
            scored_indices = sorted(
                enumerate(scores), 
                key=lambda x: -x[1]
            )[:n_results * 2]
            
            for idx, score in scored_indices:
                if score > 0:
                    bm25_results.append((
                        chunk_ids[idx],
                        score,
                        chunk_data[idx]
                    ))
        
        # 3. RRF融合
        if bm25_results:
            fused_results = self._rrf_fusion(vector_results, bm25_results, self.rrf_k)
        else:
            fused_results = vector_results
        
        return fused_results[:n_results]
    
    def retrieve_with_parent(
        self,
        query: str,
        user_id: str,
        n_results: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        检索并获取父文档上下文
        
        Args:
            query: 查询文本
            user_id: 用户ID
            n_results: 返回结果数量
            document_ids: 限定搜索的文档ID列表
            
        Returns:
            List[Dict]: 包含父文档内容的检索结果
        """
        # 先检索子切片
        child_results = self.retrieve(
            query=query,
            user_id=user_id,
            n_results=n_results,
            document_ids=document_ids
        )
        
        # 获取对应的父切片
        enriched_results = []
        seen_parents = set()
        
        for result in child_results:
            parent_id = result["metadata"].get("parent_chunk_id")
            
            if parent_id and parent_id not in seen_parents:
                parent = self.vector_store.get_chunk_by_id(parent_id)
                if parent:
                    result["parent_content"] = parent["content"]
                    result["parent_metadata"] = parent["metadata"]
                    seen_parents.add(parent_id)
            
            enriched_results.append(result)
        
        return enriched_results
