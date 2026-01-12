"""
数据库模型定义
使用SQLAlchemy ORM定义所有数据表
"""

import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, 
    DateTime, Text, ForeignKey, JSON, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

def generate_uuid() -> str:
    """生成UUID字符串"""
    return str(uuid.uuid4())


class User(Base):
    """用户表"""
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(128), nullable=False)
    display_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # 关系
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    knowledge_states = relationship("KnowledgeState", back_populates="user", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="user", cascade="all, delete-orphan")


class UserModelPreference(Base):
    """
    用户模型偏好表：支持“按模块”选择不同厂商/模型。

    说明：
    - API Key 仍由服务端环境变量（.env）统一管理，不存入数据库。
    - 该表只记录用户选择的 provider/model 等偏好。
    """

    __tablename__ = "user_model_preferences"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # 模块标识，例如：rag / kg / quiz / socratic
    module = Column(String(50), nullable=False)

    # 厂商/提供方，例如：google / siliconflow
    provider = Column(String(50), nullable=False)

    # 模型名称，例如：gemini-2.0-flash / Qwen2.5-7B-Instruct
    model = Column(String(100), nullable=False)

    # 可选：OpenAI 兼容 API base（例如硅基流动）
    api_base = Column(String(255), nullable=True)

    # 可选：该模块使用的温度
    temperature = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_user_model_pref_unique", "user_id", "module", unique=True),
    )


class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(20), nullable=False)  # pdf, docx, md
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # bytes
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    upload_time = Column(DateTime, default=datetime.utcnow)
    process_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    chunk_count = Column(Integer, default=0)
    # 使用非保留名作为属性，数据库列名仍为 metadata
    extra_metadata = Column("metadata", JSON, nullable=True)  # 存储额外元数据
    
    # 关系
    user = relationship("User", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_document_user_status", "user_id", "process_status"),
    )


class Chunk(Base):
    """文档切片表"""
    __tablename__ = "chunks"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=False, index=True)
    parent_chunk_id = Column(String(36), nullable=True, index=True)  # 父Chunk ID，用于父文档索引
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # 在文档中的顺序
    start_page = Column(Integer, nullable=True)  # 起始页码
    end_page = Column(Integer, nullable=True)  # 结束页码
    section_title = Column(String(255), nullable=True)  # 章节标题
    chunk_type = Column(String(20), default="child")  # parent, child
    token_count = Column(Integer, nullable=True)
    extra_metadata = Column("metadata", JSON, nullable=True)  # 额外元数据
    
    # 关系
    document = relationship("Document", back_populates="chunks")
    
    __table_args__ = (
        Index("idx_chunk_document_type", "document_id", "chunk_type"),
    )


class KnowledgeGraphCache(Base):
    """知识图谱缓存表（持久化到磁盘 / SQLite）"""
    __tablename__ = "knowledge_graph_cache"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # 业务唯一键（同一用户下唯一）
    graph_key = Column(String(255), nullable=False)
    scope = Column(String(20), default="document")  # document | multi_document

    # 用于缓存失效判断
    source_hash = Column(String(64), nullable=False)  # sha256 hex

    # 关联文档（用于清理/追溯）
    document_ids = Column(JSON, nullable=True)  # List[str]

    # 图谱数据（序列化后的 KnowledgeGraph dict）
    graph_data = Column(JSON, nullable=False)
    meta = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_kg_cache_user_key", "user_id", "graph_key", unique=True),
    )


class Conversation(Base):
    """对话会话表"""
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)  # 对话标题，可自动生成
    mode = Column(String(20), default="normal")  # normal, socratic, quiz
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    summary = Column(Text, nullable=True)  # 对话摘要，用于长对话记忆
    
    # 关系
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    """对话消息表"""
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    conversation_id = Column(String(36), ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    citations = Column(JSON, nullable=True)  # 引用来源 [{"doc_id": "", "chunk_id": "", "page": 1}]
    extra_metadata = Column("metadata", JSON, nullable=True)  # 额外元数据
    
    # 关系
    conversation = relationship("Conversation", back_populates="messages")
    
    __table_args__ = (
        Index("idx_message_conversation_time", "conversation_id", "created_at"),
    )


class KnowledgeState(Base):
    """用户知识状态表 - 追踪用户对各知识点的掌握情况"""
    __tablename__ = "knowledge_states"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    knowledge_point = Column(String(255), nullable=False, index=True)  # 知识点标签
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=True)  # 关联文档
    attempts = Column(Integer, default=0)  # 尝试次数
    correct_count = Column(Integer, default=0)  # 正确次数
    mastery_rate = Column(Float, default=0.0)  # 掌握率
    last_attempt = Column(DateTime, nullable=True)  # 最后尝试时间
    is_weak_point = Column(Boolean, default=False)  # 是否为薄弱点
    notes = Column(Text, nullable=True)  # 学习笔记
    
    # 关系
    user = relationship("User", back_populates="knowledge_states")
    
    __table_args__ = (
        Index("idx_knowledge_user_point", "user_id", "knowledge_point"),
        Index("idx_knowledge_weak", "user_id", "is_weak_point"),
    )


class QuizAttempt(Base):
    """测验尝试记录表"""
    __tablename__ = "quiz_attempts"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    document_id = Column(String(36), ForeignKey("documents.id"), nullable=True)
    question = Column(Text, nullable=False)
    question_type = Column(String(20), default="single")  # single, multiple
    options = Column(JSON, nullable=False)  # 选项列表
    correct_answer = Column(JSON, nullable=False)  # 正确答案
    user_answer = Column(JSON, nullable=True)  # 用户答案
    is_correct = Column(Boolean, nullable=True)
    explanation = Column(Text, nullable=True)  # 答案解析
    knowledge_points = Column(JSON, nullable=True)  # 涉及的知识点
    created_at = Column(DateTime, default=datetime.utcnow)
    answered_at = Column(DateTime, nullable=True)
    followup_question = Column(Text, nullable=True)  # 追问题目
    
    # 关系
    user = relationship("User", back_populates="quiz_attempts")
    
    __table_args__ = (
        Index("idx_quiz_user_time", "user_id", "created_at"),
    )


def init_database(database_url: str) -> sessionmaker:
    """
    初始化数据库
    
    Args:
        database_url: 数据库连接URL
        
    Returns:
        sessionmaker: 数据库会话工厂
    """
    engine = create_engine(database_url, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    return Session
