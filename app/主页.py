"""
智能学习伴侣系统 - 主入口
基于RAG的个性化学习助手
"""

import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, timedelta

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.database.crud import DatabaseManager
from backend.auth.auth_service import AuthService

# 导入 cookie 管理器
try:
    import extra_streamlit_components as stx
    COOKIE_MANAGER_AVAILABLE = True
    COOKIE_KEY = "app_user_id"
    COOKIE_WIDGET_KEY = "app_cookie_manager"
    _cookie_manager_main = None  # 模块级缓存，避免重复实例和 key 冲突
except ImportError:
    COOKIE_MANAGER_AVAILABLE = False
    COOKIE_KEY = "app_user_id"
    COOKIE_WIDGET_KEY = "app_cookie_manager"
    _cookie_manager_main = None

# 页面配置
st.set_page_config(
    page_title="智能学习伴侣",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css():
    """加载自定义CSS - Notion简约风"""
    css = """
    <style>
    /* 隐藏Streamlit默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    
    /* 简约风格 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* 时间线样式 */
    .timeline-item {
        padding: 12px 16px;
        border-left: 3px solid #e0e0e0;
        margin-left: 8px;
        margin-bottom: 8px;
        background: #fafafa;
        border-radius: 0 8px 8px 0;
    }
    .timeline-item:hover {
        border-left-color: #667eea;
        background: #f5f5ff;
    }
    .timeline-time {
        color: #888;
        font-size: 0.85em;
    }
    .timeline-content {
        margin-top: 4px;
    }
    
    /* Metric 卡片美化 */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 16px;
        border-radius: 12px;
        color: white;
    }
    div[data-testid="stMetric"] label {
        color: rgba(255,255,255,0.85) !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: white !important;
        font-size: 2rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* 按钮样式 */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def get_cookie_manager():
    """获取 Cookie 管理器实例（缓存避免重复 key）"""
    if not COOKIE_MANAGER_AVAILABLE:
        return None
    global _cookie_manager_main
    if _cookie_manager_main is None:
        _cookie_manager_main = stx.CookieManager(key=COOKIE_WIDGET_KEY)
    return _cookie_manager_main


def init_session_state():
    """初始化会话状态"""
    if "cookie_retry" not in st.session_state:
        st.session_state.cookie_retry = 0
    if "cookie_waiting" not in st.session_state:
        st.session_state.cookie_waiting = False
    if "db_manager" not in st.session_state:
        settings.ensure_directories()
        st.session_state.db_manager = DatabaseManager(str(settings.database_path))
    if "auth_service" not in st.session_state:
        st.session_state.auth_service = AuthService(st.session_state.db_manager)
    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 尝试从 cookie 恢复登录状态
    if "user" not in st.session_state:
        st.session_state.user = None
    if "cookie_login_disabled" not in st.session_state:
        st.session_state.cookie_login_disabled = False
    
    if st.session_state.user is None and COOKIE_MANAGER_AVAILABLE:
        cookie_manager = get_cookie_manager()
        if cookie_manager:
            user_id = cookie_manager.get(COOKIE_KEY)
            if user_id:
                if not st.session_state.get("cookie_login_disabled", False):
                    user = st.session_state.db_manager.get_user_by_id(user_id)
                    if user:
                        st.session_state.user = {
                            "id": user.id,
                            "username": user.username,
                            "display_name": user.display_name
                        }
                        st.session_state.cookie_retry = 0
                        st.session_state.cookie_waiting = False
                    else:
                        st.session_state.cookie_retry += 1
                        st.session_state.cookie_waiting = st.session_state.cookie_retry <= 2
            else:
                st.session_state.cookie_retry += 1
                st.session_state.cookie_waiting = st.session_state.cookie_retry <= 2


def save_login_cookie(user_id: str):
    """保存登录状态到 cookie"""
    if COOKIE_MANAGER_AVAILABLE:
        cookie_manager = get_cookie_manager()
        if cookie_manager:
            cookie_manager.set(COOKIE_KEY, user_id, expires_at=datetime.now() + timedelta(days=7))


def clear_login_cookie():
    """清除登录 cookie"""
    if not COOKIE_MANAGER_AVAILABLE:
        return
    cookie_manager = get_cookie_manager()
    if not cookie_manager:
        return
    try:
        cookie_manager.delete(COOKIE_KEY)
    except Exception:
        pass
    try:
        cookie_manager.set(COOKIE_KEY, "", expires_at=datetime.now() - timedelta(days=1))
    except Exception:
        pass


def login_page():
    """登录/注册页面"""
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h1>📚 智能学习伴侣</h1>
        <p style="color: #888; font-size: 1.1em;">基于RAG的个性化学习助手</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("用户名", key="login_username")
            password = st.text_input("密码", type="password", key="login_password")
            submitted = st.form_submit_button("登录", use_container_width=True)
            
            if submitted:
                if username and password:
                    success, message, user = st.session_state.auth_service.login(username, password)
                    if success:
                        st.session_state.cookie_login_disabled = False
                        st.session_state.user = {
                            "id": user.id,
                            "username": user.username,
                            "display_name": user.display_name
                        }
                        save_login_cookie(user.id)
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("请输入用户名和密码")
    
    with tab2:
        with st.form("register_form"):
            new_username = st.text_input("用户名", key="reg_username")
            new_password = st.text_input("密码", type="password", key="reg_password")
            confirm_password = st.text_input("确认密码", type="password", key="reg_confirm")
            display_name = st.text_input("显示名称（可选）", key="reg_display")
            submitted = st.form_submit_button("注册", use_container_width=True)
            
            if submitted:
                if new_password != confirm_password:
                    st.error("两次输入的密码不一致")
                elif new_username and new_password:
                    success, message, user = st.session_state.auth_service.register(
                        new_username, new_password, display_name
                    )
                    if success:
                        st.success(message + "，请登录")
                    else:
                        st.error(message)
                else:
                    st.warning("请填写用户名和密码")


def get_dashboard_data(user_id: str) -> dict:
    """获取仪表盘所需的所有数据"""
    db = st.session_state.db_manager
    
    # 文档数
    docs = db.get_user_documents(user_id)
    docs_count = len(docs)
    
    # 对话数
    convs = db.get_user_conversations(user_id, active_only=False)
    conv_count = len(convs)
    
    # 测验统计
    quiz_stats = db.get_quiz_statistics(user_id)
    total_quiz = quiz_stats.get("total_answered", 0)
    correct_count = quiz_stats.get("correct_count", 0)
    
    # 错题数（待复习）
    wrong_questions = []
    if hasattr(db, 'get_recent_wrong_questions'):
        wrong_questions = db.get_recent_wrong_questions(user_id, limit=10)
    wrong_count = len(wrong_questions)
    
    # 知识点统计
    knowledge_states = db.get_user_knowledge_states(user_id)
    mastered_count = sum(1 for ks in knowledge_states if ks.mastery_rate >= 0.8)
    total_kp = len(knowledge_states)
    
    # 薄弱知识点
    weak_points = db.get_user_weak_points(user_id)
    top_weak = weak_points[0].knowledge_point if weak_points else None
    
    # 最近活动（从多个来源汇总）
    activities = []
    
    # 最近上传的文档
    for doc in docs[:3]:
        upload_time = doc.upload_time
        if upload_time:
            activities.append({
                "icon": "📄",
                "time": upload_time,
                "content": f"上传了文档 **{doc.filename}**",
                "type": "document"
            })
    
    # 最近的对话（convs 按 updated_at 倒序；展示也用 updated_at，更符合“最近互动”）
    for conv in convs[:3]:
        conv_time = getattr(conv, "updated_at", None) or getattr(conv, "created_at", None)
        if conv_time:
            is_socratic = getattr(conv, "mode", "") == "socratic"
            activities.append({
                "icon": "🧠" if is_socratic else "💬",
                "time": conv_time,
                "content": "进行了一次苏格拉底对话" if is_socratic else "进行了一次问答对话",
                "type": "conversation"
            })
    
    # 最近的错题
    for wq in wrong_questions[:3]:
        wq_time = getattr(wq, "answered_at", None) or getattr(wq, "created_at", None)
        if wq_time:
            # 获取知识点
            kps = wq.knowledge_points or []
            kp_text = kps[0] if kps else "未知知识点"
            activities.append({
                "icon": "❌",
                "time": wq_time,
                "content": f"在 **{kp_text}** 相关题目中答错",
                "type": "wrong"
            })
    
    # 按时间排序，取最近5条
    activities.sort(key=lambda x: x["time"] if x["time"] else datetime.min, reverse=True)
    activities = activities[:5]
    
    # 格式化时间显示
    # 数据库时间使用 UTC（models 默认 datetime.utcnow），这里也用 UTC，避免出现“刚发生却显示几小时前”
    now = datetime.utcnow()
    for act in activities:
        t = act["time"]
        if t:
            delta = now - t
            if delta.days == 0:
                if delta.seconds < 3600:
                    act["time_str"] = f"{delta.seconds // 60} 分钟前"
                else:
                    act["time_str"] = f"{delta.seconds // 3600} 小时前"
            elif delta.days == 1:
                act["time_str"] = "昨天"
            elif delta.days < 7:
                act["time_str"] = f"{delta.days} 天前"
            else:
                act["time_str"] = t.strftime("%m-%d")
        else:
            act["time_str"] = ""
    
    return {
        "docs_count": docs_count,
        "conv_count": conv_count,
        "total_quiz": total_quiz,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "mastered_count": mastered_count,
        "total_kp": total_kp,
        "top_weak": top_weak,
        "activities": activities,
    }


def sidebar():
    """侧边栏 - 极简"""
    with st.sidebar:
        # 用户信息
        st.markdown(
            f"### 👋 {st.session_state.user['display_name']}",
        )
        st.caption("今天也是充满智慧的一天。")
        
        st.markdown("---")
        
        # 退出登录
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.cookie_login_disabled = True
            clear_login_cookie()
            st.session_state.user = None
            st.session_state.messages = []
            st.session_state.current_conversation = None
            st.rerun()


def dashboard_page():
    """仪表盘主页面 - Notion 简约风"""
    sidebar()
    
    user_id = st.session_state.user["id"]
    display_name = st.session_state.user["display_name"]
    
    # ========== 1. 顶部问候 ==========
    hour = datetime.now().hour
    if 5 <= hour < 12:
        greeting = "早上好"
        emoji = "🌅"
    elif 12 <= hour < 18:
        greeting = "下午好"
        emoji = "☀️"
    else:
        greeting = "晚上好"
        emoji = "🌙"
    
    st.title(f"{emoji} {greeting}，{display_name}")
    st.caption("欢迎回来，这是你的学习概况。")
    
    st.divider()
    
    # ========== 2. 核心数据指标 ==========
    data = get_dashboard_data(user_id)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📚 知识库文档",
            value=data["docs_count"],
        )
    
    with col2:
        st.metric(
            label="🧠 已掌握知识点",
            value=data["mastered_count"],
            delta=f"共 {data['total_kp']} 个" if data["total_kp"] > 0 else None,
        )
    
    with col3:
        st.metric(
            label="📝 累计答题",
            value=data["total_quiz"],
            delta=f"正确 {data['correct_count']}" if data["total_quiz"] > 0 else None,
        )
    
    with col4:
        st.metric(
            label="❌ 待复习错题",
            value=data["wrong_count"],
            delta="需要复习" if data["wrong_count"] > 0 else "全部掌握 ✓",
            delta_color="inverse" if data["wrong_count"] > 0 else "normal",
        )
    
    st.divider()
    
    # ========== 3. 双栏布局：最近动态 + 智能建议 ==========
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.subheader("🕒 最近足迹")
        
        if data["activities"]:
            for item in data["activities"]:
                st.markdown(f"""
                <div class="timeline-item">
                    <span style="font-size: 1.2em; margin-right: 8px;">{item['icon']}</span>
                    <span class="timeline-time">{item['time_str']}</span>
                    <div class="timeline-content">{item['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("暂无学习记录，开始你的第一步吧！")
            if st.button("📄 上传第一份文档", use_container_width=True):
                st.switch_page("pages/2_文档管理.py")
    
    with right_col:
        st.subheader("💡 智能建议")
        
        # 根据数据生成建议
        if data["docs_count"] == 0:
            st.warning("📄 **开始学习**：上传你的第一份学习资料，开启智能学习之旅。")
            if st.button("🚀 上传文档", key="cta_upload", use_container_width=True):
                st.switch_page("pages/2_文档管理.py")
        
        elif data["wrong_count"] > 0 and data["top_weak"]:
            st.error(f"🔴 **需要复习**：检测到你在 **{data['top_weak']}** 模块错误率较高，建议立即复习。")
            if st.button(f"🎯 专项特训：{data['top_weak'][:10]}...", key="cta_weak", use_container_width=True):
                st.session_state.selected_topics = [data["top_weak"]]
                st.switch_page("pages/3_测验练习.py")
        
        elif data["total_quiz"] == 0 and data["docs_count"] > 0:
            st.info("📝 **测试一下**：你还没做过测验，来检验一下学习成果吧！")
            if st.button("🚀 开始测验", key="cta_quiz", use_container_width=True):
                st.switch_page("pages/3_测验练习.py")
        
        elif data["mastered_count"] < data["total_kp"] * 0.5 and data["total_kp"] > 0:
            st.warning(f"📈 **继续加油**：你已掌握 {data['mastered_count']}/{data['total_kp']} 个知识点，继续练习可以提升更多！")
            if st.button("🚀 继续练习", key="cta_practice", use_container_width=True):
                st.switch_page("pages/3_测验练习.py")
        
        else:
            st.success("🎉 **学习状态良好**：保持这个节奏，你做得很棒！")
            if st.button("💬 去问问 AI", key="cta_chat", use_container_width=True):
                st.switch_page("pages/1_智能问答.py")
        
        # 快捷入口
        st.markdown("---")
        st.caption("快捷入口")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💬 问答", use_container_width=True):
                st.switch_page("pages/1_智能问答.py")
        with c2:
            if st.button("📝 测验", use_container_width=True):
                st.switch_page("pages/3_测验练习.py")


def main():
    """主函数"""
    load_css()
    init_session_state()
    
    # 如果 cookie 需要再尝试恢复，则重跑一次等待 CookieManager 渲染
    if st.session_state.get("cookie_waiting"):
        st.session_state.cookie_waiting = False
        st.rerun()
    
    if st.session_state.user is None:
        login_page()
    else:
        dashboard_page()


if __name__ == "__main__":
    main()
