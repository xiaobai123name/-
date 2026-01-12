"""
智能问答页面
基于RAG的文档问答功能，支持普通问答和苏格拉底对话模式
"""

import streamlit as st
from pathlib import Path
import sys
import re

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.database.crud import DatabaseManager
from backend.retrieval.vector_store import VectorStore
from backend.rag.chain import RAGChain
from backend.learning.socratic_engine import SocraticEngine

# 导入 cookie 管理器
try:
    import extra_streamlit_components as stx
    COOKIE_MANAGER_AVAILABLE = True
    COOKIE_KEY = "app_user_id"
    COOKIE_WIDGET_KEY = "app_cookie_manager"
    _cookie_manager_chat = None  # 模块级缓存
except ImportError:
    COOKIE_MANAGER_AVAILABLE = False


def get_cookie_manager():
    """获取 Cookie 管理器实例"""
    if not COOKIE_MANAGER_AVAILABLE:
        return None
    global _cookie_manager_chat
    if _cookie_manager_chat is None:
        _cookie_manager_chat = stx.CookieManager(key=COOKIE_WIDGET_KEY)
    return _cookie_manager_chat

st.set_page_config(
    page_title="智能问答 - 学习伴侣",
    page_icon="💬",
    layout="wide"
)


def init_session():
    """初始化会话"""
    # 先初始化 db_manager（后面恢复登录需要用）
    if "db_manager" not in st.session_state:
        settings.ensure_directories()
        st.session_state.db_manager = DatabaseManager(str(settings.database_path))
    
    # 尝试从 cookie 恢复登录状态
    if ("user" not in st.session_state or st.session_state.user is None) and COOKIE_MANAGER_AVAILABLE and not st.session_state.get("cookie_login_disabled", False):
        cookie_manager = get_cookie_manager()
        if cookie_manager:
            user_id = cookie_manager.get(COOKIE_KEY)
            if user_id:
                user = st.session_state.db_manager.get_user_by_id(user_id)
                if user:
                    st.session_state.user = {
                        "id": user.id,
                        "username": user.username,
                        "display_name": user.display_name
                    }
    
    if "user" not in st.session_state or st.session_state.user is None:
        st.warning("请先登录")
        st.switch_page("主页.py")
        return False
    
    # db_manager 已在上面初始化，这里跳过
    if "db_manager" not in st.session_state:
        settings.ensure_directories()
        st.session_state.db_manager = DatabaseManager(str(settings.database_path))
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = RAGChain(
            vector_store=st.session_state.vector_store,
            db_manager=st.session_state.db_manager
        )
    
    # 初始化苏格拉底引擎
    if "socratic_engine" not in st.session_state:
        st.session_state.socratic_engine = SocraticEngine(
            vector_store=st.session_state.vector_store,
            db_manager=st.session_state.db_manager
        )
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    
    if "chat_mode" not in st.session_state:
        st.session_state.chat_mode = "普通问答"

    # 普通问答偏好（仅影响普通问答模式）
    if "qa_answer_style" not in st.session_state:
        st.session_state.qa_answer_style = "教学讲解"
    if "qa_max_words" not in st.session_state:
        st.session_state.qa_max_words = 350
    if "qa_show_inline_citations" not in st.session_state:
        st.session_state.qa_show_inline_citations = False
    if "qa_enable_clarify_question" not in st.session_state:
        st.session_state.qa_enable_clarify_question = True
    
    return True


def load_conversation(conversation_id: str):
    """加载历史对话"""
    messages = st.session_state.db_manager.get_conversation_messages(conversation_id)
    st.session_state.chat_messages = [
        {
            "role": msg.role,
            "content": msg.content,
            "citations": msg.citations
        }
        for msg in messages
    ]
    st.session_state.current_conversation_id = conversation_id


def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.markdown("### 💬 智能问答")
        
        # 返回主页
        if st.button("← 返回主页", use_container_width=True):
            st.switch_page("主页.py")
        
        st.markdown("---")
        
        # 对话模式选择
        st.markdown("#### 🎯 对话模式")
        mode = st.radio(
            "选择模式",
            ["普通问答", "苏格拉底对话"],
            label_visibility="collapsed",
            index=0 if st.session_state.chat_mode == "普通问答" else 1
        )
        
        # 显示模式说明
        if mode == "苏格拉底对话":
            st.info("💡 **苏格拉底对话模式**\n\n"
                   "系统将通过引导性问题帮助你自主发现答案，"
                   "而不是直接给出结论。这种方式能加深理解和记忆。")
        
        if mode != st.session_state.chat_mode:
            st.session_state.chat_mode = mode
            # 切换模式时重置对话
            if st.session_state.chat_messages:
                st.warning("切换模式将开始新对话")

        # 普通问答：回答偏好（默认更适合初中生的教学讲解）
        if st.session_state.chat_mode == "普通问答":
            st.markdown("---")
            st.markdown("#### ✨ 回答偏好")
            style_options = ["教学讲解", "简洁直接", "更严谨（偏考试）"]
            cur_style = st.session_state.get("qa_answer_style", style_options[0])
            st.session_state.qa_answer_style = st.radio(
                "回答风格",
                style_options,
                index=style_options.index(cur_style) if cur_style in style_options else 0,
                horizontal=True,
            )
            st.session_state.qa_max_words = st.select_slider(
                "篇幅（正文）",
                options=[200, 350, 600],
                value=int(st.session_state.get("qa_max_words", 350)),
            )
            st.session_state.qa_show_inline_citations = st.checkbox(
                "正文显示引用角标（如 〔1〕）",
                value=bool(st.session_state.get("qa_show_inline_citations", False)),
                help="关闭后正文更清爽，引用仍会在下方“参考来源”里展示。",
            )
            st.session_state.qa_enable_clarify_question = st.checkbox(
                "允许在结尾追问一个澄清问题",
                value=bool(st.session_state.get("qa_enable_clarify_question", True)),
            )
        
        st.markdown("---")
        
        # 文档筛选
        st.markdown("#### 📚 文档范围")
        docs = st.session_state.db_manager.get_user_documents(
            st.session_state.user["id"],
            status="completed"
        )
        
        if docs:
            doc_options = {doc.filename: doc.id for doc in docs}
            selected_docs = st.multiselect(
                "选择文档（留空则搜索全部）",
                options=list(doc_options.keys()),
                default=[],
                label_visibility="collapsed"
            )
            st.session_state.selected_doc_ids = [doc_options[name] for name in selected_docs] if selected_docs else None
        else:
            st.info("暂无文档，请先上传")
            st.session_state.selected_doc_ids = None
        
        st.markdown("---")
        
        # 新建对话
        if st.button("🔄 新建对话", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.current_conversation_id = None
            st.rerun()
        
        st.markdown("---")
        
        # 苏格拉底对话总结按钮
        if st.session_state.chat_mode == "苏格拉底对话" and len(st.session_state.chat_messages) >= 4:
            if st.button("📝 生成学习总结", use_container_width=True):
                with st.spinner("生成总结中..."):
                    summary = st.session_state.socratic_engine.get_summary(
                        st.session_state.chat_messages,
                        st.session_state.user["id"]
                    )
                    st.session_state.socratic_summary = summary
            
            if "socratic_summary" in st.session_state:
                st.markdown("#### 📊 学习总结")
                st.markdown(st.session_state.socratic_summary)
            
            st.markdown("---")
        
        # 历史对话列表
        st.markdown("#### 📜 历史对话")
        conversations = st.session_state.db_manager.get_user_conversations(
            st.session_state.user["id"],
            active_only=True
        )
        
        if conversations:
            for conv in conversations[:10]:  # 最多显示10条历史对话
                # 生成对话标题（使用创建时间和简短标题）
                conv_title = conv.title or f"对话 {conv.created_at.strftime('%m-%d %H:%M')}"
                if len(conv_title) > 20:
                    conv_title = conv_title[:20] + "..."
                
                # 显示对话模式标签
                mode_emoji = "🎓" if conv.mode == "socratic" else "💬"
                
                # 高亮当前选中的对话
                is_current = st.session_state.current_conversation_id == conv.id
                btn_type = "primary" if is_current else "secondary"
                
                if st.button(
                    f"{mode_emoji} {conv_title}",
                    key=f"conv_{conv.id}",
                    use_container_width=True,
                    type=btn_type
                ):
                    load_conversation(conv.id)
                    st.rerun()
        else:
            st.caption("暂无历史对话")


def display_message(role: str, content: str, citations: list = None, msg_key: str = None, is_socratic: bool = False):
    """显示消息"""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    else:
        avatar = "🎓" if is_socratic else "assistant"
        with st.chat_message("assistant", avatar=avatar if is_socratic else None):
            st.markdown(content)
            
            # 显示引用 - 使用折叠框（苏格拉底模式通常不显示引用）
            if citations and not is_socratic:
                with st.expander(f"📎 参考来源 ({len(citations)}条)", expanded=False):
                    for i, citation in enumerate(citations):
                        idx = citation.get("index", i + 1)
                        filename = citation.get("filename", "未知文件")
                        page = citation.get("page")
                        chunk_content = citation.get("content", "")
                        
                        # 显示文件名和页码
                        if page:
                            st.markdown(f"**〔{idx}〕《{filename}》 第{page}页**")
                        else:
                            st.markdown(f"**〔{idx}〕《{filename}》**")
                        
                        # 原文片段折叠按钮
                        if chunk_content:
                            unique_key = f"{msg_key or 'msg'}_{idx}_{i}"
                            with st.expander("查看原文片段", expanded=False):
                                st.markdown(f"```\n{chunk_content}\n```")
                        
                        if i < len(citations) - 1:
                            st.markdown("---")


def handle_normal_qa(prompt: str):
    """处理普通问答"""
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response_placeholder = st.empty()
                
                result = st.session_state.rag_chain.query(
                    question=prompt,
                    user_id=st.session_state.user["id"],
                    conversation_id=st.session_state.current_conversation_id,
                    document_ids=st.session_state.get("selected_doc_ids"),
                    audience="初中数学",
                    answer_style=st.session_state.get("qa_answer_style", "教学讲解"),
                    max_words=int(st.session_state.get("qa_max_words", 350)),
                    enable_clarify_question=bool(st.session_state.get("qa_enable_clarify_question", True)),
                )
                
                full_response = result.answer
                citations = result.citations
                
                display_response = full_response
                if not st.session_state.get("qa_show_inline_citations", False):
                    display_response = re.sub(r"〔\d+(?:,\s*\d+)*〕", "", display_response)
                    display_response = re.sub(r"[ \t]+\n", "\n", display_response)
                    display_response = re.sub(r"\n{3,}", "\n\n", display_response).strip()

                response_placeholder.markdown(display_response)
                
                # 显示引用
                if citations:
                    with st.expander(f"📎 参考来源 ({len(citations)}条)", expanded=False):
                        for i, citation in enumerate(citations):
                            idx = citation.get("index", i + 1)
                            filename = citation.get("filename", "未知文件")
                            page = citation.get("page")
                            chunk_content = citation.get("content", "")
                            
                            if page:
                                st.markdown(f"**〔{idx}〕《{filename}》 第{page}页**")
                            else:
                                st.markdown(f"**〔{idx}〕《{filename}》**")
                            
                            if chunk_content:
                                with st.expander("查看原文片段", expanded=False):
                                    st.markdown(f"```\n{chunk_content}\n```")
                            
                            if i < len(citations) - 1:
                                st.markdown("---")
                
                # 保存到历史
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "citations": citations
                })
                
            except Exception as e:
                st.error(f"生成回答时出错: {str(e)}")
                st.info("请检查API密钥配置是否正确")


def handle_socratic_dialogue(prompt: str):
    """处理苏格拉底对话"""
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("思考引导性问题..."):
            try:
                response_placeholder = st.empty()
                progress_placeholder = st.empty()

                # 调用苏格拉底引擎
                response = st.session_state.socratic_engine.respond(
                    user_input=prompt,
                    user_id=st.session_state.user["id"],
                    conversation_history=st.session_state.chat_messages,
                    document_ids=st.session_state.get("selected_doc_ids")
                )
                
                # 显示回复
                response_placeholder.markdown(response.question)
                
                # 显示进度条
                progress_percentage = int(response.progress * 100)
                progress_placeholder.progress(
                    response.progress, 
                    text=f"💡 理解进度: {progress_percentage}%"
                )
                
                # 显示阶段信息
                phase_names = {
                    "exploration": "🔍 探索阶段",
                    "clarification": "💬 澄清阶段",
                    "deepening": "🎯 深入阶段",
                    "synthesis": "📚 综合阶段",
                    "completion": "✅ 完成阶段"
                }
                phase_name = phase_names.get(response.phase.value, "探索阶段")
                st.caption(f"当前阶段: {phase_name}")
                
                # 保存到历史
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response.question,
                    "citations": [],
                    "is_socratic": True
                })
                
                # 保存到数据库
                if st.session_state.current_conversation_id:
                    st.session_state.db_manager.add_message(
                        conversation_id=st.session_state.current_conversation_id,
                        role="assistant",
                        content=response.question
                    )
                
            except Exception as e:
                st.error(f"苏格拉底对话出错: {str(e)}")
                st.info("请检查API配置是否正确")


def main():
    """主函数"""
    if not init_session():
        return
    
    sidebar()
    
    # 标题根据模式变化
    if st.session_state.chat_mode == "苏格拉底对话":
        st.title("🎓 苏格拉底对话")
        st.markdown("通过引导性问题帮助你深入思考，自主发现答案")
    else:
        st.title("💬 智能问答")
        st.markdown("基于您的文档进行精准问答，所有回答都有据可依")
    
    # 检查是否有文档
    docs = st.session_state.db_manager.get_user_documents(
        st.session_state.user["id"],
        status="completed"
    )
    
    if not docs:
        st.warning("⚠️ 您还没有上传任何文档，请先在「文档管理」页面上传学习资料。")
        if st.button("前往上传文档"):
            st.switch_page("pages/2_文档管理.py")
        return
    
    # 显示历史消息
    is_socratic_mode = st.session_state.chat_mode == "苏格拉底对话"
    for i, msg in enumerate(st.session_state.chat_messages):
        display_message(
            msg["role"], 
            msg["content"], 
            msg.get("citations"), 
            msg_key=f"history_{i}",
            is_socratic=msg.get("is_socratic", False) or is_socratic_mode
        )
    
    # 输入提示文字
    if is_socratic_mode:
        input_placeholder = "回答问题或提出你的疑问..."
    else:
        input_placeholder = "输入您的问题..."
    
    # 用户输入
    if prompt := st.chat_input(input_placeholder):
        # 添加用户消息
        st.session_state.chat_messages.append({
            "role": "user",
            "content": prompt
        })
        display_message("user", prompt)
        
        # 创建对话（如果是新对话）
        if st.session_state.current_conversation_id is None:
            conv = st.session_state.db_manager.create_conversation(
                user_id=st.session_state.user["id"],
                mode="socratic" if is_socratic_mode else "normal"
            )
            st.session_state.current_conversation_id = conv.id
        
        # 保存用户消息到数据库
        st.session_state.db_manager.add_message(
            conversation_id=st.session_state.current_conversation_id,
            role="user",
            content=prompt
        )
        
        # 根据模式处理
        if is_socratic_mode:
            handle_socratic_dialogue(prompt)
        else:
            handle_normal_qa(prompt)


if __name__ == "__main__":
    main()
