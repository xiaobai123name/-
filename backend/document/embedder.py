"""
文档向量化模块
支持 Gemini Embedding 与 OpenAI 兼容 Embedding API
"""

from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import httpx
from google import genai
from google.genai import types as genai_types

from ..config import settings


class DocumentEmbedder:
    """文档向量化器"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        """
        初始化向量化器
        
        Args:
            api_key: API密钥（provider=google 时为 GOOGLE_API_KEY；provider=antigravity 时为 ANTIGRAVITY_API_KEY）
            model_name: 嵌入模型名称（google: gemini-embedding-001；OpenAI兼容: text-embedding-3-small 等）
            provider: Embedding 提供方：google / antigravity（默认读 settings.EMBEDDING_PROVIDER）
            api_base: OpenAI兼容的 Embeddings API Base（provider=antigravity 时可覆盖 settings.ANTIGRAVITY_API_BASE）
        """
        self.provider = (provider or getattr(settings, "EMBEDDING_PROVIDER", "google") or "google").strip().lower()
        self.model_name = (model_name or getattr(settings, "EMBEDDING_MODEL", "") or "gemini-embedding-001").strip()

        # provider-specific settings
        self.api_base = (api_base or getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip().rstrip("/")

        if self.provider == "google":
            self.api_key = api_key or settings.GOOGLE_API_KEY
            self._google_client = genai.Client(api_key=self.api_key) if self.api_key else None
        elif self.provider == "antigravity":
            self.api_key = api_key or getattr(settings, "ANTIGRAVITY_API_KEY", "")
            self._google_client = None
        else:
            raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {self.provider}")

    @staticmethod
    def _normalize_google_model_name(model_name: str) -> str:
        name = (model_name or "").strip()
        if not name:
            return "gemini-embedding-001"
        return name.removeprefix("models/")

    @staticmethod
    def _normalize_task_type(task_type: str) -> str:
        task = (task_type or "").strip().lower()
        mapping = {
            "retrieval_document": "RETRIEVAL_DOCUMENT",
            "retrieval_query": "RETRIEVAL_QUERY",
            "semantic_similarity": "SEMANTIC_SIMILARITY",
            "classification": "CLASSIFICATION",
            "clustering": "CLUSTERING",
        }
        return mapping.get(task, "RETRIEVAL_DOCUMENT")

    @staticmethod
    def _extract_google_embeddings(response: object) -> List[List[float]]:
        embeddings = getattr(response, "embeddings", None) or []
        vectors: List[List[float]] = []
        for emb in embeddings:
            values = getattr(emb, "values", None)
            if isinstance(values, list):
                vectors.append(values)
        return vectors

    def _embed_via_openai_compatible(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("未配置 Antigravity Embedding API Key（ANTIGRAVITY_API_KEY）")
        if not self.api_base:
            raise ValueError("未配置 Antigravity Embedding API Base（ANTIGRAVITY_API_BASE）")
        if not self.model_name:
            raise ValueError("未配置 Embedding 模型名称（EMBEDDING_MODEL）")

        url = f"{self.api_base}/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model_name, "input": texts}

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("data") or []
        if not isinstance(items, list):
            raise RuntimeError("Embedding API 返回格式不正确：缺少 data 列表")

        # 优先按 index 回填，避免返回顺序不稳定
        out: List[Optional[List[float]]] = [None] * len(texts)
        for it in items:
            if not isinstance(it, dict):
                continue
            idx = it.get("index")
            emb = it.get("embedding")
            if isinstance(idx, int) and 0 <= idx < len(out) and isinstance(emb, list):
                out[idx] = emb

        if any(e is None for e in out):
            # 兜底：尝试按返回顺序读取 embeddings
            seq = [it.get("embedding") for it in items if isinstance(it, dict) and isinstance(it.get("embedding"), list)]
            if len(seq) == len(texts):
                out = seq  # type: ignore[assignment]
            else:
                raise RuntimeError("Embedding API 返回 embeddings 数量不匹配")

        return [e for e in out if e is not None]
    
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

        if self.provider == "google":
            if not self.api_key:
                raise ValueError("未配置 Google Embedding API Key（GOOGLE_API_KEY）")
            if self._google_client is None:
                self._google_client = genai.Client(api_key=self.api_key)
            model = self._normalize_google_model_name(self.model_name)
            config = genai_types.EmbedContentConfig(
                task_type=self._normalize_task_type(task_type)
            )
            result = self._google_client.models.embed_content(
                model=model,
                contents=text,
                config=config,
            )
            vectors = self._extract_google_embeddings(result)
            if not vectors:
                raise RuntimeError("Google Embedding API 返回为空")
            return vectors[0]

        # OpenAI-compatible（Antigravity）
        return self._embed_via_openai_compatible([text])[0]
    
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
        
        all_embeddings: List[List[float]] = []

        # 分批处理
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]

            if self.provider == "google":
                if not self.api_key:
                    raise ValueError("未配置 Google Embedding API Key（GOOGLE_API_KEY）")
                if self._google_client is None:
                    self._google_client = genai.Client(api_key=self.api_key)
                model = self._normalize_google_model_name(self.model_name)
                config = genai_types.EmbedContentConfig(
                    task_type=self._normalize_task_type(task_type)
                )
                result = self._google_client.models.embed_content(
                    model=model,
                    contents=batch,
                    config=config,
                )
                vectors = self._extract_google_embeddings(result)
                if len(vectors) != len(batch):
                    raise RuntimeError("Google Embedding API 返回 embeddings 数量不匹配")
                all_embeddings.extend(vectors)
            else:
                all_embeddings.extend(self._embed_via_openai_compatible(batch))

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
        # gemini-embedding-001 默认 3072 维；OpenAI兼容维度取决于具体模型
        if self.provider == "google" and "gemini-embedding-001" in self._normalize_google_model_name(self.model_name):
            return 3072
        return 768
