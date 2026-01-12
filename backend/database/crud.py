"""
数据库CRUD操作管理
提供所有数据库操作的统一接口
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from contextlib import contextmanager

from .models import (
    User, Document, Chunk, Conversation, 
    Message, KnowledgeState, QuizAttempt, KnowledgeGraphCache, UserModelPreference, init_database
)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, database_path: str):
        """
        初始化数据库管理器
        
        Args:
            database_path: 数据库文件路径
        """
        self.database_url = f"sqlite:///{database_path}"
        self.Session = init_database(self.database_url)
    
    @contextmanager
    def get_session(self):
        """获取数据库会话的上下文管理器"""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    # ==================== 用户操作 ====================
    
    def create_user(self, username: str, password_hash: str, display_name: Optional[str] = None) -> User:
        """创建用户"""
        with self.get_session() as session:
            user = User(
                username=username,
                password_hash=password_hash,
                display_name=display_name or username
            )
            session.add(user)
            session.flush()
            return user
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        with self.get_session() as session:
            return session.query(User).filter(User.username == username).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根据ID获取用户"""
        with self.get_session() as session:
            return session.query(User).filter(User.id == user_id).first()
    
    def update_user_login(self, user_id: str) -> None:
        """更新用户登录时间"""
        with self.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                user.last_login = datetime.utcnow()
    
    # ==================== 文档操作 ====================
    
    def create_document(
        self, 
        user_id: str, 
        filename: str, 
        file_type: str, 
        file_path: str, 
        file_size: int,
        title: Optional[str] = None,
        description: Optional[str] = None
    ) -> Document:
        """创建文档记录"""
        with self.get_session() as session:
            doc = Document(
                user_id=user_id,
                filename=filename,
                file_type=file_type,
                file_path=file_path,
                file_size=file_size,
                title=title or filename,
                description=description
            )
            session.add(doc)
            session.flush()
            return doc
    
    def get_user_documents(self, user_id: str, status: Optional[str] = None) -> List[Document]:
        """获取用户的所有文档"""
        with self.get_session() as session:
            query = session.query(Document).filter(Document.user_id == user_id)
            if status:
                query = query.filter(Document.process_status == status)
            return query.order_by(desc(Document.upload_time)).all()
    
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        """根据ID获取文档"""
        with self.get_session() as session:
            return session.query(Document).filter(Document.id == doc_id).first()
    
    def update_document_status(self, doc_id: str, status: str, chunk_count: int = 0) -> None:
        """更新文档处理状态"""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.process_status = status
                doc.chunk_count = chunk_count
    
    def delete_document(self, doc_id: str) -> bool:
        """删除文档及其所有切片"""
        with self.get_session() as session:
            doc = session.query(Document).filter(Document.id == doc_id).first()
            if doc:
                # 同步清理知识图谱缓存（避免遗留脏数据）
                self._delete_kg_cache_by_document_in_session(session, doc.user_id, doc_id)
                session.delete(doc)
                return True
            return False

    # ==================== 知识图谱缓存（持久化） ====================

    def get_kg_cache(self, user_id: str, graph_key: str) -> Optional[KnowledgeGraphCache]:
        """获取知识图谱缓存记录"""
        with self.get_session() as session:
            return (
                session.query(KnowledgeGraphCache)
                .filter(
                    and_(
                        KnowledgeGraphCache.user_id == user_id,
                        KnowledgeGraphCache.graph_key == graph_key,
                    )
                )
                .first()
            )

    def upsert_kg_cache(
        self,
        user_id: str,
        graph_key: str,
        source_hash: str,
        graph_data: Dict[str, Any],
        document_ids: Optional[List[str]] = None,
        scope: str = "document",
        meta: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeGraphCache:
        """创建或更新知识图谱缓存记录"""
        with self.get_session() as session:
            record = (
                session.query(KnowledgeGraphCache)
                .filter(
                    and_(
                        KnowledgeGraphCache.user_id == user_id,
                        KnowledgeGraphCache.graph_key == graph_key,
                    )
                )
                .first()
            )

            if record:
                record.source_hash = source_hash
                record.graph_data = graph_data
                record.document_ids = document_ids or record.document_ids
                record.scope = scope
                record.meta = meta
                record.updated_at = datetime.utcnow()
            else:
                record = KnowledgeGraphCache(
                    user_id=user_id,
                    graph_key=graph_key,
                    scope=scope,
                    source_hash=source_hash,
                    document_ids=document_ids or [],
                    graph_data=graph_data,
                    meta=meta,
                )
                session.add(record)

            session.flush()
            return record

    def delete_kg_cache(self, user_id: str, graph_key: str) -> int:
        """删除指定 key 的知识图谱缓存"""
        with self.get_session() as session:
            deleted = (
                session.query(KnowledgeGraphCache)
                .filter(
                    and_(
                        KnowledgeGraphCache.user_id == user_id,
                        KnowledgeGraphCache.graph_key == graph_key,
                    )
                )
                .delete(synchronize_session=False)
            )
            return deleted

    def delete_kg_cache_by_document(self, user_id: str, document_id: str) -> int:
        """删除与某文档相关的所有知识图谱缓存（包含单文档与多文档缓存）"""
        with self.get_session() as session:
            return self._delete_kg_cache_by_document_in_session(session, user_id, document_id)

    def _delete_kg_cache_by_document_in_session(self, session: Session, user_id: str, document_id: str) -> int:
        """在给定 session 内删除与文档相关的 KG 缓存（内部辅助方法）"""
        # SQLite JSON 字段无法稳定使用 contains()，用 LIKE 做保守匹配（document_id 为 UUID，误匹配概率极低）
        like_pattern = f'%"{document_id}"%'
        deleted = (
            session.query(KnowledgeGraphCache)
            .filter(
                and_(
                    KnowledgeGraphCache.user_id == user_id,
                    or_(
                        KnowledgeGraphCache.graph_key.like(f"%{document_id}%"),
                        KnowledgeGraphCache.document_ids.like(like_pattern),  # type: ignore[attr-defined]
                    ),
                )
            )
            .delete(synchronize_session=False)
        )
        return deleted

    # ==================== 用户模型偏好（按模块） ====================

    def get_user_model_preferences(self, user_id: str) -> Dict[str, Dict[str, Any]]:
        """
        获取用户的所有模块模型偏好。

        Returns:
            Dict[module, {provider, model, api_base, temperature}]
        """
        with self.get_session() as session:
            rows = (
                session.query(UserModelPreference)
                .filter(UserModelPreference.user_id == user_id)
                .all()
            )
            prefs: Dict[str, Dict[str, Any]] = {}
            for r in rows:
                prefs[r.module] = {
                    "provider": r.provider,
                    "model": r.model,
                    "api_base": r.api_base,
                    "temperature": r.temperature,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
            return prefs

    def get_user_model_preference(self, user_id: str, module: str) -> Optional[UserModelPreference]:
        """获取用户某个模块的模型偏好记录（若不存在返回 None）。"""
        with self.get_session() as session:
            return (
                session.query(UserModelPreference)
                .filter(
                    and_(
                        UserModelPreference.user_id == user_id,
                        UserModelPreference.module == module,
                    )
                )
                .first()
            )

    def upsert_user_model_preference(
        self,
        user_id: str,
        module: str,
        provider: str,
        model: str,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> UserModelPreference:
        """创建或更新用户某模块的模型偏好。"""
        provider = (provider or "").strip().lower()
        module = (module or "").strip().lower()
        model = (model or "").strip()

        with self.get_session() as session:
            record = (
                session.query(UserModelPreference)
                .filter(
                    and_(
                        UserModelPreference.user_id == user_id,
                        UserModelPreference.module == module,
                    )
                )
                .first()
            )

            if record:
                record.provider = provider
                record.model = model
                record.api_base = api_base
                record.temperature = temperature
                record.updated_at = datetime.utcnow()
            else:
                record = UserModelPreference(
                    user_id=user_id,
                    module=module,
                    provider=provider,
                    model=model,
                    api_base=api_base,
                    temperature=temperature,
                )
                session.add(record)

            session.flush()
            return record
    
    # ==================== 切片操作 ====================
    
    def create_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[Chunk]:
        """批量创建文档切片"""
        with self.get_session() as session:
            chunks = [Chunk(**data) for data in chunks_data]
            session.add_all(chunks)
            session.flush()
            return chunks
    
    def get_document_chunks(self, doc_id: str, chunk_type: Optional[str] = None) -> List[Chunk]:
        """获取文档的所有切片"""
        with self.get_session() as session:
            query = session.query(Chunk).filter(Chunk.document_id == doc_id)
            if chunk_type:
                query = query.filter(Chunk.chunk_type == chunk_type)
            return query.order_by(Chunk.chunk_index).all()
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Chunk]:
        """根据ID获取切片"""
        with self.get_session() as session:
            return session.query(Chunk).filter(Chunk.id == chunk_id).first()
    
    def get_parent_chunk(self, child_chunk_id: str) -> Optional[Chunk]:
        """获取子切片对应的父切片"""
        with self.get_session() as session:
            child = session.query(Chunk).filter(Chunk.id == child_chunk_id).first()
            if child and child.parent_chunk_id:
                return session.query(Chunk).filter(Chunk.id == child.parent_chunk_id).first()
            return None
    
    # ==================== 对话操作 ====================
    
    def create_conversation(self, user_id: str, mode: str = "normal", title: Optional[str] = None) -> Conversation:
        """创建对话会话"""
        with self.get_session() as session:
            conv = Conversation(
                user_id=user_id,
                mode=mode,
                title=title or f"对话 {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            session.add(conv)
            session.flush()
            return conv
    
    def get_user_conversations(self, user_id: str, active_only: bool = True) -> List[Conversation]:
        """获取用户的对话列表"""
        with self.get_session() as session:
            query = session.query(Conversation).filter(Conversation.user_id == user_id)
            if active_only:
                query = query.filter(Conversation.is_active == True)
            return query.order_by(desc(Conversation.updated_at)).all()
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str, 
        citations: Optional[List[Dict]] = None
    ) -> Message:
        """添加对话消息"""
        with self.get_session() as session:
            msg = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                citations=citations
            )
            session.add(msg)
            # 更新对话的更新时间
            conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
            if conv:
                conv.updated_at = datetime.utcnow()
            session.flush()
            return msg
    
    def get_conversation_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Message]:
        """获取对话的所有消息"""
        with self.get_session() as session:
            query = session.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at)
            if limit:
                query = query.limit(limit)
            return query.all()
    
    # ==================== 知识状态操作 ====================
    
    def get_or_create_knowledge_state(self, user_id: str, knowledge_point: str) -> KnowledgeState:
        """获取或创建知识状态记录"""
        with self.get_session() as session:
            return self._get_or_create_knowledge_state(session, user_id, knowledge_point)

    def _get_or_create_knowledge_state(self, session: Session, user_id: str, knowledge_point: str) -> KnowledgeState:
        """在给定 session 内获取或创建知识状态记录（内部辅助方法）"""
        state = session.query(KnowledgeState).filter(
            and_(
                KnowledgeState.user_id == user_id,
                KnowledgeState.knowledge_point == knowledge_point
            )
        ).first()

        if not state:
            state = KnowledgeState(
                user_id=user_id,
                knowledge_point=knowledge_point
            )
            session.add(state)
            session.flush()

        return state
    
    def update_knowledge_state(
        self, 
        user_id: str, 
        knowledge_point: str, 
        is_correct: bool
    ) -> KnowledgeState:
        """更新知识状态（答题后调用）"""
        with self.get_session() as session:
            state = self._get_or_create_knowledge_state(session, user_id, knowledge_point)
            state.attempts += 1
            if is_correct:
                state.correct_count += 1
            state.mastery_rate = state.correct_count / state.attempts
            state.last_attempt = datetime.utcnow()
            state.is_weak_point = state.mastery_rate < 0.6 and state.attempts >= 3
            session.flush()
            return state
    
    def get_user_weak_points(self, user_id: str) -> List[KnowledgeState]:
        """获取用户的薄弱知识点"""
        with self.get_session() as session:
            return session.query(KnowledgeState).filter(
                and_(
                    KnowledgeState.user_id == user_id,
                    KnowledgeState.is_weak_point == True
                )
            ).order_by(KnowledgeState.mastery_rate).all()
    
    def get_user_knowledge_states(self, user_id: str) -> List[KnowledgeState]:
        """获取用户所有知识状态"""
        with self.get_session() as session:
            return session.query(KnowledgeState).filter(
                KnowledgeState.user_id == user_id
            ).order_by(desc(KnowledgeState.last_attempt)).all()
    
    # ==================== 测验记录操作 ====================
    
    def create_quiz_attempt(
        self,
        user_id: str,
        question: str,
        question_type: str,
        options: List[str],
        correct_answer: List[str],
        explanation: str,
        knowledge_points: List[str],
        document_id: Optional[str] = None
    ) -> QuizAttempt:
        """创建测验题目"""
        with self.get_session() as session:
            attempt = QuizAttempt(
                user_id=user_id,
                document_id=document_id,
                question=question,
                question_type=question_type,
                options=options,
                correct_answer=correct_answer,
                explanation=explanation,
                knowledge_points=knowledge_points
            )
            session.add(attempt)
            session.flush()
            return attempt
    
    def submit_quiz_answer(
        self, 
        attempt_id: str, 
        user_answer: List[str],
        followup_question: Optional[str] = None
    ) -> QuizAttempt:
        """提交测验答案"""
        with self.get_session() as session:
            attempt = session.query(QuizAttempt).filter(QuizAttempt.id == attempt_id).first()
            if attempt:
                attempt.user_answer = user_answer
                attempt.is_correct = set(user_answer) == set(attempt.correct_answer)
                attempt.answered_at = datetime.utcnow()
                attempt.followup_question = followup_question
                session.flush()
            return attempt
    
    def get_user_quiz_history(self, user_id: str, limit: int = 50) -> List[QuizAttempt]:
        """获取用户测验历史"""
        with self.get_session() as session:
            return session.query(QuizAttempt).filter(
                QuizAttempt.user_id == user_id
            ).order_by(desc(QuizAttempt.created_at)).limit(limit).all()
    
    def get_unanswered_quiz_attempts(self, user_id: str) -> List[QuizAttempt]:
        """获取用户未回答的测验题目"""
        with self.get_session() as session:
            return session.query(QuizAttempt).filter(
                and_(
                    QuizAttempt.user_id == user_id,
                    QuizAttempt.user_answer == None
                )
            ).order_by(QuizAttempt.created_at).all()
    
    def get_quiz_attempt_by_id(self, attempt_id: str) -> Optional[QuizAttempt]:
        """根据ID获取测验题目"""
        with self.get_session() as session:
            return session.query(QuizAttempt).filter(QuizAttempt.id == attempt_id).first()
    
    def get_quiz_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取用户测验统计数据"""
        with self.get_session() as session:
            # 获取所有已回答的测验
            answered_quizzes = session.query(QuizAttempt).filter(
                and_(
                    QuizAttempt.user_id == user_id,
                    QuizAttempt.user_answer != None
                )
            ).all()
            
            total = len(answered_quizzes)
            correct = sum(1 for q in answered_quizzes if q.is_correct)
            
            return {
                "total_answered": total,
                "correct_count": correct,
                "accuracy": correct / total if total > 0 else 0.0
            }
    
    def delete_unanswered_quiz_attempts(self, user_id: str) -> int:
        """删除用户未回答的测验题目（用于重新开始）"""
        with self.get_session() as session:
            deleted = session.query(QuizAttempt).filter(
                and_(
                    QuizAttempt.user_id == user_id,
                    QuizAttempt.user_answer == None
                )
            ).delete()
            return deleted
    
    def get_answered_quiz_history(self, user_id: str, limit: int = 20) -> List[QuizAttempt]:
        """获取用户已回答的测验历史（用于历史记录显示）"""
        with self.get_session() as session:
            return session.query(QuizAttempt).filter(
                and_(
                    QuizAttempt.user_id == user_id,
                    QuizAttempt.user_answer != None
                )
            ).order_by(desc(QuizAttempt.answered_at)).limit(limit).all()
    
    def get_quiz_session_history(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户的测验会话历史（按时间分组）
        返回每次测验的汇总信息
        """
        with self.get_session() as session:
            # 获取所有已回答的测验
            all_quizzes = session.query(QuizAttempt).filter(
                and_(
                    QuizAttempt.user_id == user_id,
                    QuizAttempt.user_answer != None
                )
            ).order_by(desc(QuizAttempt.answered_at)).all()
            
            if not all_quizzes:
                return []
            
            # 按创建时间分组（同一批次的测验创建时间相近）
            sessions = []
            current_session = []
            last_time = None
            
            for quiz in all_quizzes:
                if last_time is None:
                    current_session = [quiz]
                    last_time = quiz.created_at
                else:
                    # 如果创建时间差距小于60秒，认为是同一批次
                    time_diff = abs((quiz.created_at - last_time).total_seconds())
                    if time_diff < 60:
                        current_session.append(quiz)
                    else:
                        # 保存当前会话
                        if current_session:
                            sessions.append(current_session)
                        current_session = [quiz]
                    last_time = quiz.created_at
            
            # 保存最后一个会话
            if current_session:
                sessions.append(current_session)
            
            # 转换为汇总信息
            result = []
            for i, quizzes in enumerate(sessions):
                total = len(quizzes)
                correct = sum(1 for q in quizzes if q.is_correct)
                answered_at = quizzes[0].answered_at if quizzes[0].answered_at else quizzes[0].created_at
                
                result.append({
                    "session_id": i,
                    "quiz_ids": [q.id for q in quizzes],
                    "total_questions": total,
                    "correct_count": correct,
                    "accuracy": correct / total if total > 0 else 0,
                    "answered_at": answered_at,
                    "quizzes": quizzes  # 保留完整数据
                })
            
            return result

    # ==================== 测验专项/错题本支持 ====================

    def get_recent_wrong_questions(self, user_id: str, limit: int = 5) -> List[QuizAttempt]:
        """获取最近的错题（用于错题本 Top N 展示）"""
        with self.get_session() as session:
            return (
                session.query(QuizAttempt)
                .filter(
                    and_(
                        QuizAttempt.user_id == user_id,
                        QuizAttempt.user_answer != None,  # noqa: E711
                        QuizAttempt.is_correct == False,  # noqa: E712
                    )
                )
                .order_by(desc(QuizAttempt.answered_at))
                .limit(limit)
                .all()
            )

    def get_topic_error_stats(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取知识点错题统计与掌握情况。

        Returns:
            List[Dict]:
                - knowledge_point: str
                - wrong_count: int
                - mastery_rate: Optional[float]
                - attempts: int
                - correct_count: int
                - is_weak_point: bool
        """
        with self.get_session() as session:
            # 1) 读取用户已有知识点集合（用于补全“未练习”知识点）
            states = session.query(KnowledgeState).filter(KnowledgeState.user_id == user_id).all()
            state_points = {s.knowledge_point for s in states}

            # 2) 从 quiz_attempts 计算每个知识点的练习次数/正确次数/错题数（更贴近真实答题记录）
            answered_attempts = (
                session.query(QuizAttempt)
                .filter(
                    and_(
                        QuizAttempt.user_id == user_id,
                        QuizAttempt.user_answer != None,  # noqa: E711
                    )
                )
                .all()
            )

            attempts_counts: Dict[str, int] = defaultdict(int)
            correct_counts: Dict[str, int] = defaultdict(int)
            wrong_counts: Dict[str, int] = defaultdict(int)

            for attempt in answered_attempts:
                kps = attempt.knowledge_points or []
                for kp in kps:
                    if not kp:
                        continue
                    attempts_counts[kp] += 1
                    if attempt.is_correct:
                        correct_counts[kp] += 1
                    else:
                        wrong_counts[kp] += 1

            # 3) 合并知识点集合
            all_points = state_points | set(attempts_counts.keys()) | set(wrong_counts.keys())
            stats: List[Dict[str, Any]] = []
            for kp in all_points:
                attempts = int(attempts_counts.get(kp, 0))
                correct = int(correct_counts.get(kp, 0))
                wrong = int(wrong_counts.get(kp, 0))
                mastery_rate = (correct / attempts) if attempts > 0 else None

                stats.append(
                    {
                        "knowledge_point": kp,
                        "wrong_count": wrong,
                        "mastery_rate": mastery_rate,
                        "attempts": attempts,
                        "correct_count": correct,
                        "is_weak_point": bool(attempts >= 3 and mastery_rate is not None and mastery_rate < 0.6),
                    }
                )

            # 4) 排序：错题多优先，其次掌握率低优先
            def _sort_key(item: Dict[str, Any]):
                mastery = item["mastery_rate"]
                mastery_val = mastery if mastery is not None else 1.0
                return (-item["wrong_count"], mastery_val, item["knowledge_point"])

            stats.sort(key=_sort_key)
            return stats

    def mark_question_solved(
        self,
        attempt_id: str,
        user_answer: Optional[List[str]] = None,
    ) -> Optional[QuizAttempt]:
        """
        标记错题为已解决：写回用户答案与正确性，并同步修正 knowledge_states 的正确次数。

        说明：
        - 该方法主要用于“错题本重做”。
        - 若原记录为错题且本次答对，会将其从错题本移除（is_correct=True）。
        """
        with self.get_session() as session:
            attempt = session.query(QuizAttempt).filter(QuizAttempt.id == attempt_id).first()
            if not attempt:
                return None

            prev_is_correct = attempt.is_correct
            new_answer = user_answer if user_answer is not None else (attempt.correct_answer or [])
            new_is_correct = set(new_answer) == set(attempt.correct_answer or [])

            attempt.user_answer = new_answer
            attempt.is_correct = new_is_correct
            attempt.answered_at = datetime.utcnow()
            session.flush()

            # 仅在“由错变对”时，修正知识点统计（避免双计）
            if prev_is_correct is False and new_is_correct:
                for kp in (attempt.knowledge_points or []):
                    if not kp:
                        continue
                    state = (
                        session.query(KnowledgeState)
                        .filter(
                            and_(
                                KnowledgeState.user_id == attempt.user_id,
                                KnowledgeState.knowledge_point == kp,
                            )
                        )
                        .first()
                    )

                    if state:
                        # 尝试将一次错误“纠正”为正确（不增加 attempts）
                        if state.correct_count < state.attempts:
                            state.correct_count += 1
                        else:
                            # 兜底：如果数据不一致，就按一次新的正确练习处理
                            state.attempts += 1
                            state.correct_count += 1

                        state.mastery_rate = (
                            state.correct_count / state.attempts if state.attempts > 0 else 1.0
                        )
                        state.last_attempt = datetime.utcnow()
                        state.is_weak_point = state.mastery_rate < 0.6 and state.attempts >= 3
                    else:
                        # 没有知识状态就补一条（按一次正确练习）
                        session.add(
                            KnowledgeState(
                                user_id=attempt.user_id,
                                knowledge_point=kp,
                                attempts=1,
                                correct_count=1,
                                mastery_rate=1.0,
                                last_attempt=datetime.utcnow(),
                                is_weak_point=False,
                            )
                        )

                session.flush()

            return attempt