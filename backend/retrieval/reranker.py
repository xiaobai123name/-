"""
自适应重排序模块
实现基于置信度的自适应重排序策略
"""

import httpx
from typing import List, Dict, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings


class AdaptiveReranker:
    """自适应重排序器"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "BAAI/bge-reranker-v2-m3",
        confidence_threshold: float = 0.15,
        api_base: str = "https://api.siliconflow.cn/v1"
    ):
        """
        初始化重排序器
        
        Args:
            api_key: 硅基流动API密钥
            model: 重排序模型名称
            confidence_threshold: 置信度阈值（用于自适应策略）
            api_base: API基础URL
        """
        self.api_key = api_key or settings.SILICONFLOW_API_KEY
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.api_base = api_base
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        计算检索结果的置信度（Top1与Top2的分差）
        
        Args:
            results: 检索结果列表
            
        Returns:
            float: 置信度分数
        """
        if len(results) < 2:
            return 1.0  # 只有一个结果时，置信度最高
        
        # 获取分数字段（可能是score、fusion_score或rrf_score）
        score_key = "fusion_score" if "fusion_score" in results[0] else "score"
        
        top1_score = results[0].get(score_key, 0)
        top2_score = results[1].get(score_key, 0)
        
        return top1_score - top2_score
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_rerank_api(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        调用硅基流动重排序API
        
        Args:
            query: 查询文本
            documents: 待重排序的文档列表
            top_n: 返回top-n结果
            
        Returns:
            List[Dict]: 重排序结果
        """
        if not self.api_key:
            raise ValueError("未配置硅基流动API密钥")
        
        url = f"{self.api_base}/rerank"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
            "return_documents": True
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
        return response.json().get("results", [])
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回top-k结果
            
        Returns:
            List[Dict]: 重排序后的结果
        """
        if not results:
            return []
        
        if not self.api_key:
            # 没有API密钥，返回原结果
            return results[:top_k]
        
        # 提取文档内容
        documents = [r["content"] for r in results]
        
        try:
            # 调用重排序API
            rerank_results = self._call_rerank_api(query, documents, top_k)
            
            # 根据重排序结果重新排序原始结果
            reranked = []
            for item in rerank_results:
                idx = item["index"]
                original = results[idx].copy()
                original["rerank_score"] = item["relevance_score"]
                reranked.append(original)
            
            return reranked
        except Exception as e:
            print(f"重排序失败，使用原始排序: {e}")
            return results[:top_k]
    
    def adaptive_rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        自适应重排序：根据置信度决定是否执行重排序
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回top-k结果
            
        Returns:
            Tuple[List[Dict], bool]: (重排序后的结果, 是否执行了重排序)
        """
        if not results:
            return [], False
        
        # 计算置信度
        confidence = self._calculate_confidence(results)
        
        if confidence >= self.confidence_threshold:
            # 高置信度：跳过重排序
            return results[:top_k], False
        else:
            # 低置信度：执行重排序
            reranked = self.rerank(query, results, top_k)
            return reranked, True
    
    def get_retrieval_stats(
        self,
        results: List[Dict[str, Any]],
        was_reranked: bool
    ) -> Dict[str, Any]:
        """
        获取检索统计信息
        
        Args:
            results: 检索结果
            was_reranked: 是否执行了重排序
            
        Returns:
            Dict: 统计信息
        """
        if not results:
            return {
                "result_count": 0,
                "was_reranked": was_reranked,
                "confidence": 0
            }
        
        confidence = self._calculate_confidence(results)
        
        stats = {
            "result_count": len(results),
            "was_reranked": was_reranked,
            "confidence": confidence,
            "top_score": results[0].get("rerank_score") or results[0].get("fusion_score") or results[0].get("score", 0)
        }
        
        return stats
