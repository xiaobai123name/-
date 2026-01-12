"""
智能测验生成器
基于文档内容自动生成测验题目
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..config import settings
from ..retrieval.vector_store import VectorStore
from ..retrieval.hybrid_retriever import HybridRetriever
from ..database.crud import DatabaseManager
from ..llm.router import ModelRouter
from ..rag.prompts import PromptTemplates


@dataclass
class QuizQuestion:
    """测验题目数据结构"""
    id: int
    question_type: str  # "single" or "multiple"
    question: str
    options: List[str]
    correct_answer: List[str]
    explanation: str
    knowledge_points: List[str]
    difficulty: str  # "easy", "medium", "hard"


@dataclass
class FollowupQuestion:
    """追问题目数据结构"""
    question: str
    options: List[str]
    correct_answer: List[str]
    hint: str
    error_analysis: str
    simplified_explanation: str


class QuizGenerator:
    """测验生成器"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        db_manager: DatabaseManager,
        api_key: Optional[str] = None
    ):
        """
        初始化测验生成器
        
        Args:
            vector_store: 向量存储实例
            db_manager: 数据库管理器
            api_key: Google API密钥
        """
        self.vector_store = vector_store
        self.db = db_manager
        self.retriever = HybridRetriever(vector_store)
        # 模型路由：按用户+模块动态选择 provider/model
        self.model_router = ModelRouter(db_manager)
    
    def _parse_json_response(self, response: str) -> Dict:
        """
        解析LLM返回的JSON响应
        
        Args:
            response: LLM响应文本
            
        Returns:
            Dict: 解析后的JSON对象
        """
        # 尝试提取JSON块
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接解析
            json_str = response
        
        # 清理可能的问题
        json_str = json_str.strip()
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # 尝试修复常见问题
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
    
    def generate_quiz(
        self,
        user_id: str,
        document_ids: Optional[List[str]] = None,
        num_questions: int = 5,
        question_type: str = "mixed",  # "single", "multiple", "mixed"
        difficulty: str = "medium",
        knowledge_points: Optional[List[str]] = None
    ) -> List[QuizQuestion]:
        """
        生成测验题目
        
        Args:
            user_id: 用户ID
            document_ids: 文档ID列表（为空则使用所有文档）
            num_questions: 题目数量
            question_type: 题型 ("single", "multiple", "mixed")
            difficulty: 难度 ("easy", "medium", "hard")
            knowledge_points: 指定的知识点列表
            
        Returns:
            List[QuizQuestion]: 生成的题目列表
        """
        # 1. 检索相关文档内容
        # 如果有指定知识点，用知识点作为查询
        if knowledge_points:
            query = " ".join(knowledge_points)
        else:
            query = "核心概念 重要知识点 关键定义"
        
        chunks = self.retriever.retrieve(
            query=query,
            user_id=user_id,
            n_results=10,
            document_ids=document_ids
        )
        
        if not chunks:
            return []
        
        # 2. 格式化上下文
        context = "\n\n".join([chunk["content"] for chunk in chunks])
        
        # 3. 确定题型描述
        if question_type == "single":
            type_desc = "单选"
        elif question_type == "multiple":
            type_desc = "多选"
        else:
            type_desc = "单选和多选混合（比例约7:3）"
        
        # 4. 构建Prompt并调用LLM
        prompt = PromptTemplates.get_quiz_prompt()
        messages = prompt.format_messages(
            context=context,
            num_questions=num_questions,
            question_type=type_desc,
            difficulty=difficulty,
        )

        llm = self.model_router.get_chat_model(user_id=user_id, module="quiz", streaming=False)
        response = llm.invoke(messages)
        
        # 5. 解析响应
        try:
            result = self._parse_json_response(response.content)
            questions_data = result.get("questions", [])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析测验题目失败: {e}")
            return []
        
        # 6. 转换为QuizQuestion对象
        questions = []
        for i, q_data in enumerate(questions_data):
            try:
                question = QuizQuestion(
                    id=i + 1,
                    question_type=q_data.get("type", "single"),
                    question=q_data["question"],
                    options=q_data["options"],
                    correct_answer=q_data["correct_answer"],
                    explanation=q_data.get("explanation", ""),
                    knowledge_points=q_data.get("knowledge_points", []),
                    difficulty=q_data.get("difficulty", difficulty)
                )
                questions.append(question)
            except KeyError:
                continue
        
        return questions
    
    def generate_followup(
        self,
        original_question: QuizQuestion,
        user_answer: List[str],
        user_id: str,
        document_ids: Optional[List[str]] = None
    ) -> Optional[FollowupQuestion]:
        """
        生成错误追问题目
        
        Args:
            original_question: 原始题目
            user_answer: 用户的错误答案
            user_id: 用户ID
            document_ids: 文档ID列表
            
        Returns:
            Optional[FollowupQuestion]: 追问题目
        """
        # 1. 检索相关知识点的文档内容
        query = original_question.question + " " + " ".join(original_question.knowledge_points)
        chunks = self.retriever.retrieve(
            query=query,
            user_id=user_id,
            n_results=5,
            document_ids=document_ids
        )
        
        context = "\n\n".join([chunk["content"] for chunk in chunks]) if chunks else "无相关文档内容"
        
        # 2. 构建Prompt
        prompt = PromptTemplates.get_followup_prompt()
        messages = prompt.format_messages(
            original_question=original_question.question,
            options="\n".join(original_question.options),
            user_answer=", ".join(user_answer),
            correct_answer=", ".join(original_question.correct_answer),
            explanation=original_question.explanation,
            context=context
        )

        llm = self.model_router.get_chat_model(user_id=user_id, module="quiz", streaming=False)
        response = llm.invoke(messages)
        
        # 3. 解析响应
        try:
            result = self._parse_json_response(response.content)
            followup_data = result.get("followup_question", {})
            
            return FollowupQuestion(
                question=followup_data["question"],
                options=followup_data["options"],
                correct_answer=followup_data["correct_answer"],
                hint=followup_data.get("hint", ""),
                error_analysis=result.get("error_analysis", ""),
                simplified_explanation=result.get("simplified_explanation", "")
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析追问题目失败: {e}")
            return None
    
    def save_quiz_attempt(
        self,
        user_id: str,
        question: QuizQuestion,
        document_id: Optional[str] = None
    ) -> str:
        """
        保存测验题目到数据库
        
        Args:
            user_id: 用户ID
            question: 题目对象
            document_id: 文档ID
            
        Returns:
            str: 题目记录ID
        """
        attempt = self.db.create_quiz_attempt(
            user_id=user_id,
            question=question.question,
            question_type=question.question_type,
            options=question.options,
            correct_answer=question.correct_answer,
            explanation=question.explanation,
            knowledge_points=question.knowledge_points,
            document_id=document_id
        )
        return attempt.id
    
    def submit_answer(
        self,
        attempt_id: str,
        user_answer: List[str],
        generate_followup: bool = True,
        user_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[FollowupQuestion]]:
        """
        提交答案并处理结果
        
        Args:
            attempt_id: 题目记录ID
            user_answer: 用户答案
            generate_followup: 答错时是否生成追问
            user_id: 用户ID（用于生成追问）
            document_ids: 文档ID列表（用于生成追问）
            
        Returns:
            Tuple[bool, Optional[FollowupQuestion]]: (是否正确, 追问题目)
        """
        # 获取题目记录
        # 这里需要从数据库获取题目信息
        # 简化处理：直接提交答案
        attempt = self.db.submit_quiz_answer(attempt_id, user_answer)
        
        is_correct = attempt.is_correct
        followup = None
        
        # 如果答错且需要生成追问
        if not is_correct and generate_followup and user_id:
            # 重建QuizQuestion对象
            original = QuizQuestion(
                id=0,
                question_type=attempt.question_type,
                question=attempt.question,
                options=attempt.options,
                correct_answer=attempt.correct_answer,
                explanation=attempt.explanation,
                knowledge_points=attempt.knowledge_points or [],
                difficulty="medium"
            )
            
            followup = self.generate_followup(
                original_question=original,
                user_answer=user_answer,
                user_id=user_id,
                document_ids=document_ids
            )
            
            # 保存追问题目
            if followup:
                self.db.submit_quiz_answer(
                    attempt_id,
                    user_answer,
                    followup_question=followup.question
                )
        
        return is_correct, followup
