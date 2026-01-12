"""
配置管理模块
使用pydantic-settings进行类型安全的配置管理
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类"""
    
    # === 基础路径 ===
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    # === Google Gemini API ===
    GOOGLE_API_KEY: str = Field(default="", description="Google Gemini API密钥")
    
    # === LlamaParse API ===
    LLAMA_CLOUD_API_KEY: str = Field(default="", description="LlamaParse API密钥")
    
    # === 硅基流动 API ===
    SILICONFLOW_API_KEY: str = Field(default="", description="硅基流动 API密钥")

    # === 硅基流动 LLM（OpenAI 兼容）配置 ===
    SILICONFLOW_LLM_TIMEOUT_SEC: float = Field(default=60.0, description="硅基流动 Chat Completions 超时（秒）")
    SILICONFLOW_LLM_MAX_TOKENS: int = Field(default=1024, description="硅基流动 Chat Completions 输出最大 tokens")
    
    # === 数据库配置 ===
    DATABASE_PATH: str = Field(default="data/learning_companion.db")
    
    # === ChromaDB配置 ===
    CHROMA_PERSIST_DIR: str = Field(default="data/chroma_db")
    
    # === 文档上传配置 ===
    UPLOAD_DIR: str = Field(default="data/uploads")
    MAX_UPLOAD_SIZE_MB: int = Field(default=50)
    
    # === 模型配置 ===
    LLM_MODEL: str = Field(default="gemini-2.0-flash")
    EMBEDDING_MODEL: str = Field(default="text-embedding-004")
    
    # === RAG配置 ===
    CHUNK_SIZE: int = Field(default=500, description="子Chunk大小")
    CHUNK_OVERLAP: int = Field(default=50, description="Chunk重叠大小")
    PARENT_CHUNK_SIZE: int = Field(default=2000, description="父Chunk大小")
    TOP_K_RETRIEVAL: int = Field(default=10, description="检索返回数量")
    RERANK_TOP_K: int = Field(default=5, description="重排序后返回数量")
    CONFIDENCE_THRESHOLD: float = Field(default=0.15, description="置信度阈值")

    # === Knowledge Graph 配置 ===
    KG_MAX_CONCURRENCY: int = Field(default=3, description="知识图谱构建最大并发数")
    KG_MIN_REQUEST_INTERVAL_SEC: float = Field(default=4.0, description="知识图谱 LLM 请求最小间隔（秒）")
    KG_EXTRACTION_MAX_CHARS: int = Field(default=8000, description="知识图谱抽取输入文本最大长度（字符）")
    KG_MERGE_TARGET_CHARS: int = Field(default=4500, description="语义合并目标长度（字符）")
    KG_MERGE_MAX_CHARS: int = Field(default=7500, description="语义合并最大长度（字符）")
    KG_SEMANTIC_MERGE_THRESHOLD: float = Field(default=0.75, description="语义合并相似度阈值")
    KG_EMBED_BATCH_SIZE: int = Field(default=64, description="知识图谱 embedding 分批大小")
    KG_ALIGNMENT_DIRECT_THRESHOLD: float = Field(default=0.80, description="实体对齐：直接合并相似度阈值")
    KG_ALIGNMENT_LLM_THRESHOLD: float = Field(default=0.70, description="实体对齐：触发 LLM 判断相似度阈值")
    KG_ALIGNMENT_MAX_LLM_CHECKS: int = Field(default=50, description="实体对齐：最多 LLM 判断次数")
    KG_ENABLE_PERSISTENT_CACHE: bool = Field(default=True, description="是否启用知识图谱 SQLite 持久化缓存")

    # SiliconFlow 作为 KG 抽取模型时的额外参数（避免大输出导致超时）
    KG_LLM_TIMEOUT_SEC: float = Field(default=180.0, description="知识图谱 LLM 调用超时（秒）")
    KG_LLM_MAX_TOKENS: int = Field(default=1024, description="知识图谱 LLM 输出最大 tokens")
    
    # === 应用配置 ===
    APP_TITLE: str = Field(default="智能学习伴侣")
    APP_LANGUAGE: str = Field(default="zh-CN")
    DEBUG: bool = Field(default=False)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    @property
    def database_path(self) -> Path:
        """获取数据库完整路径"""
        return self.BASE_DIR / self.DATABASE_PATH
    
    @property
    def chroma_path(self) -> Path:
        """获取ChromaDB完整路径"""
        return self.BASE_DIR / self.CHROMA_PERSIST_DIR
    
    @property
    def upload_path(self) -> Path:
        """获取上传目录完整路径"""
        return self.BASE_DIR / self.UPLOAD_DIR
    
    def ensure_directories(self) -> None:
        """确保必要的目录存在"""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.upload_path.mkdir(parents=True, exist_ok=True)


# 全局配置单例
settings = Settings()
