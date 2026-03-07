"""
测验练习页面
智能生成测验题目，追踪学习进度
"""

import streamlit as st
from pathlib import Path
import sys
import importlib
import re
from typing import Dict, List

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.database.crud import DatabaseManager
from backend.retrieval.vector_store import VectorStore
from backend.learning.quiz_generator import QuizGenerator, QuizQuestion
from backend.learning.knowledge_tracker import KnowledgeTracker
from app.auth_cookie import ensure_auth_state_defaults, restore_user_from_cookie


def _ensure_db_manager_class():
    """
    Streamlit 会缓存已 import 的模块；当我们给 DatabaseManager 新增方法后，老进程里仍可能是旧类。
    这里在运行时按需 reload `backend.database.crud`，确保拿到包含新方法的 DatabaseManager。
    """
    global DatabaseManager

    required = ("get_topic_error_stats", "get_recent_wrong_questions", "mark_question_solved")
    if all(hasattr(DatabaseManager, m) for m in required):
        return DatabaseManager

    try:
        from backend.database import crud as crud_module

        crud_module = importlib.reload(crud_module)
        DatabaseManager = crud_module.DatabaseManager
    except Exception:
        # reload 失败则保持原样（后续会以 AttributeError 的形式提示）
        pass

    return DatabaseManager

st.set_page_config(
    page_title="测验练习 - 学习伴侣",
    page_icon="📝",
    layout="wide"
)


def init_session():
    """初始化会话"""
    ensure_auth_state_defaults()

    # 确保使用最新的 DatabaseManager（热更新后旧实例可能缺少新方法）
    needs_new_db = (
        "db_manager" not in st.session_state
        or not hasattr(st.session_state.db_manager, "get_unanswered_quiz_attempts")
        or not hasattr(st.session_state.db_manager, "get_quiz_session_history")
        or not hasattr(st.session_state.db_manager, "get_topic_error_stats")
        or not hasattr(st.session_state.db_manager, "get_recent_wrong_questions")
        or not hasattr(st.session_state.db_manager, "mark_question_solved")
    )
    if needs_new_db:
        settings.ensure_directories()
        Db = _ensure_db_manager_class()
        st.session_state.db_manager = Db(str(settings.database_path))
        # 关联的 knowledge_tracker 也需要使用新的 db_manager
        st.session_state.knowledge_tracker = KnowledgeTracker(st.session_state.db_manager)
    
    restore_status = restore_user_from_cookie(st.session_state.db_manager)
    if restore_status == "pending":
        st.info("正在恢复登录状态...")
        st.stop()
    
    if "user" not in st.session_state or st.session_state.user is None:
        st.warning("请先登录")
        st.switch_page("主页.py")
        return False
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if "quiz_generator" not in st.session_state:
        st.session_state.quiz_generator = QuizGenerator(
            vector_store=st.session_state.vector_store,
            db_manager=st.session_state.db_manager
        )
    
    if "knowledge_tracker" not in st.session_state:
        st.session_state.knowledge_tracker = KnowledgeTracker(st.session_state.db_manager)
    
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    
    if "followup_question" not in st.session_state:
        st.session_state.followup_question = None
    
    # 从数据库加载未完成的测验（持久化）
    if "current_quiz" not in st.session_state or st.session_state.current_quiz is None:
        unanswered = st.session_state.db_manager.get_unanswered_quiz_attempts(
            st.session_state.user["id"]
        )
        if unanswered:
            # 转换为 QuizQuestion 对象
            st.session_state.current_quiz = []
            st.session_state.quiz_attempt_ids = []  # 存储数据库ID
            for attempt in unanswered:
                question = QuizQuestion(
                    id=len(st.session_state.current_quiz) + 1,
                    question_type=attempt.question_type,
                    question=attempt.question,
                    options=attempt.options,
                    correct_answer=attempt.correct_answer,
                    explanation=attempt.explanation or "",
                    knowledge_points=attempt.knowledge_points or [],
                    difficulty="medium"
                )
                st.session_state.current_quiz.append(question)
                st.session_state.quiz_attempt_ids.append(attempt.id)
        else:
            st.session_state.current_quiz = None
            st.session_state.quiz_attempt_ids = []
    
    return True


def load_quiz_history(session_data: dict):
    """加载历史测验记录"""
    quizzes = session_data["quizzes"]
    st.session_state.current_quiz = []
    st.session_state.quiz_attempt_ids = []
    st.session_state.quiz_answers = {}
    
    for i, quiz in enumerate(quizzes):
        question = QuizQuestion(
            id=i + 1,
            question_type=quiz.question_type,
            question=quiz.question,
            options=quiz.options,
            correct_answer=quiz.correct_answer,
            explanation=quiz.explanation or "",
            knowledge_points=quiz.knowledge_points or [],
            difficulty="medium"
        )
        st.session_state.current_quiz.append(question)
        st.session_state.quiz_attempt_ids.append(quiz.id)
        
        # 恢复用户的历史答案
        if quiz.user_answer:
            st.session_state.quiz_answers[i] = quiz.user_answer
    
    # 标记为已提交（查看历史模式）
    st.session_state.quiz_submitted = True
    st.session_state.viewing_history = True


def _truncate(text: str, max_len: int = 36) -> str:
    """截断文本用于侧边栏摘要展示"""
    if not text:
        return ""
    s = " ".join(text.split())
    return s if len(s) <= max_len else s[: max_len - 1] + "…"








def render_explanation(
    *,
    options: List[str],
    correct_answer: List[str],
    user_answer: List[str],
    explanation: str,
    question_type: str,
    show_mistake_hint: bool,
):
    """以“老师讲题”风格渲染解析：先结论+错因，再表格对比，最后原文折叠。"""
    option_letters = [opt.split(".")[0].strip() for opt in options if opt]
    reasons = _extract_option_reasons(explanation, option_letters)
    is_correct = set(user_answer or []) == set(correct_answer or [])

    st.markdown("### ✅ 核心结论")
    st.markdown(
        f"- 你的答案：**{', '.join(user_answer) if user_answer else '（未作答）'}**\n"
        f"- 正确答案：**{', '.join(correct_answer)}**"
    )

    if (not is_correct) and show_mistake_hint:
        st.warning(_build_mistake_hint(user_answer, correct_answer, question_type))

    st.markdown("### 🧾 逐项判定（对比表）")
    rows = []
    for opt in options:
        letter = opt.split(".")[0].strip()
        is_true = letter in (correct_answer or [])
        verdict = "✅ 正确" if is_true else "❌ 错误"

        reason = reasons.get(letter, "")
        if not reason:
            reason = "它符合题干条件对应的性质。" if is_true else "它把相近性质混为一谈，条件不满足。"

        rows.append((letter, verdict, reason))

    table_md = "| 选项 | 判定 | 一句话理由 |\n|---|---|---|\n" + "\n".join(
        [f"| {a} | {b} | {c} |" for a, b, c in rows]
    )
    st.markdown(table_md)

    if explanation:
        with st.expander("📚 原解析/依据（可选展开）", expanded=False):
            st.markdown(explanation)


def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.markdown("### 📝 测验练习")
        
        if st.button("← 返回主页", use_container_width=True):
            st.switch_page("主页.py")
        
        st.markdown("---")

        # 兼容：老会话里的 db_manager 可能是旧类实例（缺少新方法），这里做一次兜底热修复
        required_db_methods = (
            "get_topic_error_stats",
            "get_recent_wrong_questions",
            "mark_question_solved",
        )
        if any(not hasattr(st.session_state.db_manager, m) for m in required_db_methods):
            Db = _ensure_db_manager_class()
            st.session_state.db_manager = Db(str(settings.database_path))
            st.session_state.knowledge_tracker = KnowledgeTracker(st.session_state.db_manager)
        
        # ==================== 学习统计（仅保留两项） ====================
        quiz_stats = st.session_state.db_manager.get_quiz_statistics(st.session_state.user["id"])
        st.markdown("#### 📊 学习统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("累计答题数", quiz_stats["total_answered"])
        with col2:
            st.metric("正确率", f"{quiz_stats['accuracy'] * 100:.0f}%")

        st.markdown("---")

        # ==================== 智能专项选择器 ====================
        st.markdown("#### 🎯 智能专项选择器")
        if "selected_topics" not in st.session_state:
            st.session_state.selected_topics = []

        topic_stats = st.session_state.db_manager.get_topic_error_stats(st.session_state.user["id"])

        if not topic_stats:
            st.caption("暂无知识点数据，先做一组测验后这里会自动出现。")
        else:
            topics = [x["knowledge_point"] for x in topic_stats]
            topic_label_map = {}
            weak_topics = []
            topic_stats_map = {x["knowledge_point"]: x for x in topic_stats}

            for x in topic_stats:
                kp = x["knowledge_point"]
                wrong = int(x.get("wrong_count", 0))
                mastery = x.get("mastery_rate")
                attempts = int(x.get("attempts", 0))
                is_weak = bool(x.get("is_weak_point")) or (mastery is not None and mastery < 0.6) or wrong > 0
                short_kp = _truncate(kp, 16)

                # 统一：attempts==0 视为“未练习”（不依赖 mastery_rate 默认值）
                if attempts == 0:
                    icon = "⚪"
                    label = f"{icon} (未练习) {short_kp}"
                elif wrong > 0:
                    icon = "🔴"
                    label = f"{icon} (错{wrong}) {short_kp}"
                else:
                    pct = f"{(float(mastery) * 100):.0f}%" if mastery is not None else "—"
                    if mastery is not None and mastery >= 0.8:
                        icon = "🟢"
                    elif mastery is not None and mastery >= 0.4:
                        icon = "🟡"
                    else:
                        icon = "🔴"
                    label = f"{icon} (掌握{pct}) {short_kp}"

                topic_label_map[kp] = label
                if is_weak:
                    weak_topics.append(kp)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ 全选薄弱项", use_container_width=True):
                    st.session_state.selected_topics = weak_topics
                    st.rerun()
            with c2:
                if st.button("🧹 清空", use_container_width=True):
                    st.session_state.selected_topics = []
                    st.rerun()

            st.multiselect(
                "选择知识点",
                options=topics,
                key="selected_topics",
                format_func=lambda t: topic_label_map.get(t, t),
            )

            # 选中项的完整信息展示区（解决下拉项被截断导致状态/名称看不全的问题）
            if st.session_state.selected_topics:
                with st.expander("已选知识点（完整信息）", expanded=False):
                    for kp in st.session_state.selected_topics:
                        x = topic_stats_map.get(kp, {})
                        wrong = int(x.get("wrong_count", 0))
                        attempts = int(x.get("attempts", 0))
                        correct = int(x.get("correct_count", 0))
                        mastery = x.get("mastery_rate")

                        if attempts == 0:
                            mastery_text = "未练习"
                        else:
                            mastery_text = f"{(float(mastery) * 100):.0f}%" if mastery is not None else "—"

                        st.write(f"- {kp}  | 错{wrong}  | 正确{correct}/{attempts}  | 掌握{mastery_text}")

        st.markdown("---")

        # ==================== 错题本（Top 5 + st.dialog 重做） ====================
        st.markdown("#### 🧾 错题本（Top 5）")

        wrong_questions = st.session_state.db_manager.get_recent_wrong_questions(
            st.session_state.user["id"], limit=5
        )

        @st.dialog("错题重做")
        def _redo_dialog(attempt_id: str):
            attempt = st.session_state.db_manager.get_quiz_attempt_by_id(attempt_id)
            if not attempt:
                st.error("未找到题目记录")
                return

            q_type = "单选" if attempt.question_type == "single" else "多选"
            st.subheader(f"🧩 错题重做（{q_type}）")
            st.markdown(attempt.question)

            if attempt.knowledge_points:
                st.caption("相关知识点：" + " | ".join(attempt.knowledge_points))

            user_answer = []
            if attempt.question_type == "single":
                selected = st.radio(
                    "选择答案",
                    options=attempt.options,
                    key=f"redo_{attempt_id}_single",
                )
                if selected:
                    user_answer = [selected.split(".")[0].strip()]
            else:
                st.markdown("*（多选题，可选择多个答案）*")
                chosen = []
                for idx, opt in enumerate(attempt.options):
                    if st.checkbox(opt, key=f"redo_{attempt_id}_opt_{idx}"):
                        chosen.append(opt.split(".")[0].strip())
                user_answer = chosen

            if st.button("✅ 提交重做", type="primary", use_container_width=True):
                if not user_answer:
                    st.warning("请先选择答案")
                    return

                is_correct = set(user_answer) == set(attempt.correct_answer or [])
                if is_correct:
                    st.session_state.db_manager.mark_question_solved(attempt_id, user_answer=user_answer)
                    st.toast("✅ 已更新：错题已移除", icon="✅")
                    st.rerun()

                st.error(f"❌ 不正确。正确答案: {', '.join(attempt.correct_answer or [])}")
                if attempt.explanation:
                    with st.expander("📖 查看解析", expanded=True):
                        st.markdown(attempt.explanation)

        if not wrong_questions:
            st.caption("暂无错题")
        else:
            for q in wrong_questions:
                title = _truncate(q.question, 40)
                if st.button(f"❌ {title}", key=f"wrong_{q.id}", use_container_width=True):
                    _redo_dialog(q.id)

        st.markdown("---")

        # ==================== 其他（收起） ====================
        with st.expander("📜 答题历史", expanded=False):
            if hasattr(st.session_state.db_manager, "get_quiz_session_history"):
                quiz_history = st.session_state.db_manager.get_quiz_session_history(
                    st.session_state.user["id"]
                )

                if quiz_history:
                    for session_data in quiz_history[:10]:  # 最多显示10条历史
                        answered_at = session_data["answered_at"]
                        total = session_data["total_questions"]
                        correct = session_data["correct_count"]
                        accuracy_pct = session_data["accuracy"] * 100

                        time_str = answered_at.strftime("%m-%d %H:%M") if answered_at else "未知时间"
                        title = f"📋 {time_str} ({correct}/{total}题 {accuracy_pct:.0f}%)"

                        is_current = (
                            st.session_state.get("viewing_history")
                            and st.session_state.get("current_history_ids") == session_data["quiz_ids"]
                        )
                        btn_type = "primary" if is_current else "secondary"

                        if st.button(
                            title,
                            key=f"history_{session_data['session_id']}",
                            use_container_width=True,
                            type=btn_type,
                        ):
                            st.session_state.current_history_ids = session_data["quiz_ids"]
                            load_quiz_history(session_data)
                            st.rerun()
                else:
                    st.caption("暂无答题历史")
            else:
                st.caption("请刷新页面加载历史功能")

        if st.button("🔄 重新开始", use_container_width=True):
            st.session_state.db_manager.delete_unanswered_quiz_attempts(st.session_state.user["id"])
            st.session_state.current_quiz = None
            st.session_state.quiz_attempt_ids = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.followup_question = None
            st.session_state.viewing_history = False
            st.rerun()


def generate_quiz_section():
    """生成测验的设置界面"""
    st.markdown("### ⚙️ 测验设置")

    def _start_quiz(document_ids, num_questions, question_type, difficulty, knowledge_points=None):
        with st.spinner("正在生成测验题目..."):
            try:
                questions = st.session_state.quiz_generator.generate_quiz(
                    user_id=st.session_state.user["id"],
                    document_ids=document_ids,
                    num_questions=num_questions,
                    question_type=question_type,
                    difficulty=difficulty,
                    knowledge_points=knowledge_points,
                )

                if not questions:
                    st.error("生成测验失败，请重试（或检查文档是否已处理完成）")
                    return

                st.session_state.quiz_attempt_ids = []
                for question in questions:
                    attempt_id = st.session_state.quiz_generator.save_quiz_attempt(
                        user_id=st.session_state.user["id"],
                        question=question,
                        document_id=document_ids[0] if document_ids else None,
                    )
                    st.session_state.quiz_attempt_ids.append(attempt_id)

                st.session_state.current_quiz = questions
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.viewing_history = False
                st.session_state.followup_question = None
                st.rerun()

            except Exception as e:
                st.error(f"生成测验时出错: {str(e)}")

    tab1, tab2 = st.tabs(["📚 综合复习", "🎯 专项突破"])

    with tab1:
        st.caption("基于选定的文档范围出题")

        docs = st.session_state.db_manager.get_user_documents(
            st.session_state.user["id"], status="completed"
        )

        if not docs:
            st.warning("请先上传并处理完成文档后再生成测验")
            if st.button("前往上传文档", key="goto_upload_docs"):
                st.switch_page("pages/2_文档管理.py")
            return

        doc_options = {doc.filename: doc.id for doc in docs}
        selected_docs = st.multiselect(
            "选择文档范围",
            options=list(doc_options.keys()),
            default=list(doc_options.keys())[:1] if doc_options else [],
            key="review_selected_docs",
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            num_questions = st.slider(
                "题目数量", min_value=3, max_value=10, value=5, key="review_num_questions"
            )
        with c2:
            question_type = st.selectbox(
                "题目类型",
                options=["mixed", "single", "multiple"],
                key="review_question_type",
                format_func=lambda x: {"mixed": "混合题型", "single": "单选题", "multiple": "多选题"}[x],
            )
        with c3:
            difficulty = st.selectbox(
                "难度",
                options=["easy", "medium", "hard"],
                key="review_difficulty",
                format_func=lambda x: {"easy": "简单", "medium": "中等", "hard": "困难"}[x],
            )

        if st.button("🎯 生成测验", type="primary", use_container_width=True, key="review_start"):
            if not selected_docs:
                st.warning("请至少选择一个文档")
            else:
                doc_ids = [doc_options[name] for name in selected_docs]
                _start_quiz(
                    document_ids=doc_ids,
                    num_questions=num_questions,
                    question_type=question_type,
                    difficulty=difficulty,
                )

    with tab2:
        st.caption("基于侧边栏选择的知识点出题（自动跨文档检索）")

        selected_topics = st.session_state.get("selected_topics", [])
        if selected_topics:
            st.info("🎯 当前专项知识点：" + " | ".join(selected_topics))
        else:
            st.warning("请先在侧边栏「智能专项选择器」选择至少一个知识点")

        c1, c2, c3 = st.columns(3)
        with c1:
            num_questions = st.slider(
                "题目数量", min_value=3, max_value=10, value=5, key="focus_num_questions"
            )
        with c2:
            question_type = st.selectbox(
                "题目类型",
                options=["mixed", "single", "multiple"],
                key="focus_question_type",
                format_func=lambda x: {"mixed": "混合题型", "single": "单选题", "multiple": "多选题"}[x],
            )
        with c3:
            difficulty = st.selectbox(
                "难度",
                options=["easy", "medium", "hard"],
                key="focus_difficulty",
                format_func=lambda x: {"easy": "简单", "medium": "中等", "hard": "困难"}[x],
            )

        if st.button(
            "开始专项特训",
            type="primary",
            use_container_width=True,
            key="focus_start",
            disabled=not bool(selected_topics),
        ):
            if not selected_topics:
                st.warning("请先选择知识点")
            else:
                _start_quiz(
                    document_ids=None,
                    num_questions=num_questions,
                    question_type=question_type,
                    difficulty=difficulty,
                    knowledge_points=selected_topics,
                )


def display_question(question: QuizQuestion, index: int, show_result: bool = False):
    """显示单个题目"""
    q_type = "单选" if question.question_type == "single" else "多选"
    
    # 使用 Streamlit 原生组件，避免样式冲突
    st.subheader(f"题目 {index + 1} ({q_type})")
    
    st.markdown(question.question)
    
    # 选项
    if question.question_type == "single":
        selected = st.radio(
            f"选择答案 (题目{index + 1})",
            options=question.options,
            key=f"q_{index}",
            label_visibility="collapsed",
            disabled=show_result
        )
        if selected:
            # 提取选项字母 (如 "A. xxx" -> "A")
            answer_letter = selected.split(".")[0].strip()
            st.session_state.quiz_answers[index] = [answer_letter]
    else:
        # 多选题
        st.markdown("*（多选题，可选择多个答案）*")
        selected_options = []
        for opt in question.options:
            if st.checkbox(opt, key=f"q_{index}_{opt}", disabled=show_result):
                answer_letter = opt.split(".")[0].strip()
                selected_options.append(answer_letter)
        st.session_state.quiz_answers[index] = selected_options
    
    # 显示结果
    if show_result:
        user_answer = st.session_state.quiz_answers.get(index, [])
        is_correct = set(user_answer) == set(question.correct_answer)
        
        if is_correct:
            st.success("✅ 回答正确！")
        else:
            st.error(f"❌ 回答错误。正确答案: {', '.join(question.correct_answer)}")
            
            # 显示解析
            with st.expander("📖 查看解析"):
                st.markdown(question.explanation)
        
        # 显示知识点标签
        if question.knowledge_points:
            st.markdown("**相关知识点:** " + " | ".join(question.knowledge_points))


def display_history_question(question: QuizQuestion, index: int, user_answer: list):
    """显示历史记录中的题目（只读模式）"""
    q_type = "单选" if question.question_type == "single" else "多选"
    
    st.subheader(f"题目 {index + 1} ({q_type})")
    st.markdown(question.question)
    
    # 显示选项和用户答案
    for opt in question.options:
        opt_letter = opt.split(".")[0].strip()
        is_user_selected = opt_letter in user_answer
        is_correct_answer = opt_letter in question.correct_answer
        
        if is_user_selected and is_correct_answer:
            st.markdown(f"✅ **{opt}**")
        elif is_user_selected and not is_correct_answer:
            st.markdown(f"❌ ~~{opt}~~")
        elif is_correct_answer:
            st.markdown(f"✓ {opt} *(正确答案)*")
        else:
            st.markdown(f"　 {opt}")
    
    # 显示结果
    is_correct = set(user_answer) == set(question.correct_answer)
    if is_correct:
        st.success("✅ 回答正确！")
    else:
        st.error(f"❌ 回答错误。你的答案: {', '.join(user_answer)}，正确答案: {', '.join(question.correct_answer)}")
    
    # 显示解析
    if question.explanation:
        with st.expander("📖 查看解析", expanded=not is_correct):
            st.markdown(question.explanation)
    
    # 显示知识点
    if question.knowledge_points:
        st.markdown("**相关知识点:** " + " | ".join(question.knowledge_points))


def quiz_section():
    """测验答题界面"""
    questions = st.session_state.current_quiz
    is_viewing_history = st.session_state.get("viewing_history", False)
    
    if is_viewing_history:
        # 查看历史记录模式
        st.markdown(f"### 📜 历史答题记录 ({len(questions)} 道题)")
        st.info("📋 这是您之前的答题记录，可以回顾题目和解析")
    else:
        st.markdown(f"### 📝 测验进行中 ({len(questions)} 道题)")
    
    # 进度条 / 得分统计
    if is_viewing_history or st.session_state.quiz_submitted:
        correct_count = sum(
            1 for i, q in enumerate(questions)
            if set(st.session_state.quiz_answers.get(i, [])) == set(q.correct_answer)
        )
        st.progress(correct_count / len(questions))
        st.markdown(f"得分: {correct_count} / {len(questions)} ({correct_count/len(questions)*100:.0f}%)")
    else:
        answered = len([a for a in st.session_state.quiz_answers.values() if a])
        st.progress(answered / len(questions))
        st.markdown(f"已回答: {answered} / {len(questions)}")
    
    st.markdown("---")
    
    # 显示所有题目
    if is_viewing_history:
        # 历史记录模式：使用增强的显示
        for i, question in enumerate(questions):
            user_answer = st.session_state.quiz_answers.get(i, [])
            display_history_question(question, i, user_answer)
            st.markdown("---")
        
        # 返回按钮
        if st.button("📝 开始新测验", use_container_width=True):
            st.session_state.current_quiz = None
            st.session_state.quiz_attempt_ids = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.viewing_history = False
            st.session_state.current_history_ids = None
            st.rerun()
    else:
        # 正常答题模式
        for i, question in enumerate(questions):
            display_question(question, i, st.session_state.quiz_submitted)
            st.markdown("---")
        
        # 提交按钮
        if not st.session_state.quiz_submitted:
            answered = len([a for a in st.session_state.quiz_answers.values() if a])
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📤 提交答案", type="primary", use_container_width=True):
                    if answered < len(questions):
                        st.warning("请回答所有题目后再提交")
                    else:
                        # 将答案保存到数据库（持久化）
                        attempt_ids = getattr(st.session_state, 'quiz_attempt_ids', [])
                        for i, question in enumerate(questions):
                            user_answer = st.session_state.quiz_answers.get(i, [])
                            is_correct = set(user_answer) == set(question.correct_answer)
                            
                            # 保存答案到数据库
                            if i < len(attempt_ids):
                                st.session_state.db_manager.submit_quiz_answer(
                                    attempt_ids[i],
                                    user_answer
                                )
                            
                            # 更新知识点状态
                            kps = question.knowledge_points or ["通用知识"]
                            for kp in kps:
                                st.session_state.knowledge_tracker.update_knowledge_state(
                                    user_id=st.session_state.user["id"],
                                    knowledge_point=kp,
                                    is_correct=is_correct
                                )
                        
                        st.session_state.quiz_submitted = True
                        st.rerun()
            with col2:
                if st.button("🔄 重新开始", use_container_width=True):
                    # 删除数据库中未回答的测验
                    st.session_state.db_manager.delete_unanswered_quiz_attempts(
                        st.session_state.user["id"]
                    )
                    st.session_state.current_quiz = None
                    st.session_state.quiz_attempt_ids = []
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()
        else:
            # 显示总结
            correct_count = sum(
                1 for i, q in enumerate(questions)
                if set(st.session_state.quiz_answers.get(i, [])) == set(q.correct_answer)
            )
            
            # 使用 Streamlit 原生组件显示结果
            st.success(f"🎉 测验完成！得分: {correct_count} / {len(questions)} ({correct_count/len(questions)*100:.0f}%)")
            
            if st.button("📝 开始新测验", use_container_width=True):
                st.session_state.current_quiz = None
                st.session_state.quiz_attempt_ids = []
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()


def main():
    """主函数"""
    if not init_session():
        return
    
    sidebar()
    
    st.title("📝 测验练习")
    st.markdown("基于您的学习资料自动生成测验题目，巩固所学知识")
    
    if st.session_state.current_quiz is None:
        generate_quiz_section()
    else:
        quiz_section()


if __name__ == "__main__":
    main()
