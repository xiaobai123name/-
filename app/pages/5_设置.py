"""
设置页面
系统配置、API密钥管理、用户信息
"""

import streamlit as st
from pathlib import Path
import sys
import os

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.database.crud import DatabaseManager
from backend.auth.auth_service import AuthService
from app.auth_cookie import ensure_auth_state_defaults, restore_user_from_cookie

st.set_page_config(
    page_title="设置 - 学习伴侣",
    page_icon="⚙️",
    layout="wide"
)


def init_session():
    """初始化会话"""
    ensure_auth_state_defaults()

    if "db_manager" not in st.session_state:
        settings.ensure_directories()
        st.session_state.db_manager = DatabaseManager(str(settings.database_path))

    restore_status = restore_user_from_cookie(st.session_state.db_manager)
    if restore_status == "pending":
        st.info("正在恢复登录状态...")
        st.stop()

    if "user" not in st.session_state or st.session_state.user is None:
        st.warning("请先登录")
        st.switch_page("主页.py")
        return False
    
    if "auth_service" not in st.session_state:
        st.session_state.auth_service = AuthService(st.session_state.db_manager)
    
    return True


def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.markdown("### ⚙️ 设置")
        
        if st.button("← 返回主页", use_container_width=True):
            st.switch_page("主页.py")


def api_settings_section():
    """API设置部分"""
    st.markdown("### 🔑 API 配置")
    st.markdown("配置系统所需的API密钥。密钥存储在本地 `.env` 文件中。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Google Gemini API")
        gemini_key = st.text_input(
            "API 密钥",
            value="*" * 20 if settings.GOOGLE_API_KEY else "",
            type="password",
            key="gemini_key",
            help="用于LLM对话和文本嵌入"
        )
        gemini_status = "✅ 已配置" if settings.GOOGLE_API_KEY else "❌ 未配置"
        st.markdown(f"状态: {gemini_status}")
    
    with col2:
        st.markdown("#### LlamaParse API")
        llama_key = st.text_input(
            "API 密钥",
            value="*" * 20 if settings.LLAMA_CLOUD_API_KEY else "",
            type="password",
            key="llama_key",
            help="用于高级PDF解析（可选）"
        )
        llama_status = "✅ 已配置" if settings.LLAMA_CLOUD_API_KEY else "⚠️ 未配置（使用基础解析）"
        st.markdown(f"状态: {llama_status}")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 硅基流动 API")
        sf_key = st.text_input(
            "API 密钥",
            value="*" * 20 if settings.SILICONFLOW_API_KEY else "",
            type="password",
            key="sf_key",
            help="用于文档重排序（可选）"
        )
        sf_status = "✅ 已配置" if settings.SILICONFLOW_API_KEY else "⚠️ 未配置（跳过重排序）"
        st.markdown(f"状态: {sf_status}")
    
    with col4:
        st.markdown("#### Antigravity 反代（OpenAI 兼容）")
        antigravity_base = st.text_input(
            "API Base",
            value=(getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip(),
            key="antigravity_base",
            help="从 Antigravity Tools 的“快速接入/Quick Integration”复制 base_url（通常以 /v1 或 /api/v1 结尾）。",
        )
        antigravity_key = st.text_input(
            "API Key",
            value="*" * 20 if getattr(settings, "ANTIGRAVITY_API_KEY", "") else "",
            type="password",
            key="antigravity_key",
            help="用于通过 Antigravity 反代调用 Gemini/Claude 等模型（OpenAI 兼容）。",
        )
        ag_ok = bool(getattr(settings, "ANTIGRAVITY_API_KEY", "")) and bool((getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip())
        ag_status = "✅ 已配置" if ag_ok else "⚠️ 未配置"
        st.markdown(f"状态: {ag_status}")

    st.markdown("#### 配置说明")
    st.markdown(
        """
请在项目根目录创建 `.env` 文件，并填入以下内容：
```
GOOGLE_API_KEY=your_key
ANTIGRAVITY_API_KEY=your_key
ANTIGRAVITY_API_BASE=https://your-antigravity-domain/api/v1
LLAMA_CLOUD_API_KEY=your_key
SILICONFLOW_API_KEY=your_key
```
"""
    )
    
    st.info("💡 修改API密钥后需要重启应用才能生效")


def user_settings_section():
    """用户设置部分"""
    st.markdown("### 👤 用户信息")
    
    user_info = st.session_state.auth_service.get_user_info(st.session_state.user["id"])
    
    if user_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**用户名:** {user_info['username']}")
            st.markdown(f"**显示名称:** {user_info['display_name']}")
        
        with col2:
            st.markdown(f"**注册时间:** {user_info['created_at'][:10] if user_info['created_at'] else '未知'}")
            st.markdown(f"**最后登录:** {user_info['last_login'][:10] if user_info['last_login'] else '未知'}")
    
    st.markdown("---")
    
    # 修改密码
    st.markdown("#### 🔐 修改密码")
    
    with st.form("change_password"):
        old_password = st.text_input("当前密码", type="password")
        new_password = st.text_input("新密码", type="password")
        confirm_password = st.text_input("确认新密码", type="password")
        
        submitted = st.form_submit_button("修改密码")
        
        if submitted:
            if not old_password or not new_password:
                st.error("请填写所有字段")
            elif new_password != confirm_password:
                st.error("两次输入的新密码不一致")
            else:
                success, message = st.session_state.auth_service.change_password(
                    st.session_state.user["id"],
                    old_password,
                    new_password
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)


def rag_settings_section():
    """RAG设置部分"""
    st.markdown("### 🔧 RAG 参数")
    st.markdown("调整文档处理和检索的参数（高级设置）")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 分块设置")
        st.number_input(
            "子Chunk大小",
            value=settings.CHUNK_SIZE,
            min_value=100,
            max_value=2000,
            step=100,
            help="较小的值提高检索精度，较大的值保留更多上下文"
        )
        st.number_input(
            "Chunk重叠",
            value=settings.CHUNK_OVERLAP,
            min_value=0,
            max_value=200,
            step=10,
            help="相邻切片的重叠字符数"
        )
    
    with col2:
        st.markdown("#### 检索设置")
        st.number_input(
            "检索数量 (Top-K)",
            value=settings.TOP_K_RETRIEVAL,
            min_value=3,
            max_value=20,
            help="每次检索返回的文档片段数量"
        )
        st.number_input(
            "重排序数量",
            value=settings.RERANK_TOP_K,
            min_value=1,
            max_value=10,
            help="重排序后保留的片段数量"
        )
    
    with col3:
        st.markdown("#### 模型设置")
        st.text_input(
            "LLM模型",
            value=settings.LLM_MODEL,
            help="Gemini模型名称"
        )
        st.text_input(
            "Embedding模型",
            value=settings.EMBEDDING_MODEL,
            help="文本嵌入模型名称"
        )
    
    st.info("💡 参数修改需要更新 `.env` 文件并重启应用")


def model_preferences_section():
    """模型选择（按用户 + 按模块）"""
    st.markdown("### 🧠 模型选择（按模块）")
    st.markdown("为当前账号配置不同模块使用的 **厂商/模型**。API Key 仍由服务器本地 `.env` 统一管理。")

    user_id = st.session_state.user["id"]
    prefs = st.session_state.db_manager.get_user_model_preferences(user_id)

    provider_options = {
        "google": "Google Gemini",
        "siliconflow": "硅基流动 SiliconFlow（OpenAI 兼容）",
        "antigravity": "Antigravity 反代（OpenAI 兼容）",
    }

    module_defs = [
        ("rag", "智能问答（RAG）", 0.3),
        ("socratic", "苏格拉底对话", 0.7),
        ("quiz", "测验练习", 0.7),
        ("kg", "知识图谱构建", 0.3),
    ]

    st.info("💡 保存后一般 **立即生效**（无需重启）。但如果你修改了 `.env` 里的 API Key，仍需重启应用。")

    for module_key, module_name, default_temp in module_defs:
        st.markdown(f"#### {module_name}")
        current = prefs.get(module_key, {}) or {}

        # provider
        provider_default = (current.get("provider") or "google").strip().lower()
        if provider_default not in provider_options:
            provider_default = "google"
        provider = st.selectbox(
            "厂商",
            options=list(provider_options.keys()),
            index=list(provider_options.keys()).index(provider_default),
            format_func=lambda x: provider_options.get(x, x),
            key=f"model_pref_provider_{module_key}",
        )

        # model
        if provider in {"google", "antigravity"}:
            model_default = (current.get("model") or settings.LLM_MODEL or "").strip()
        else:
            model_default = (current.get("model") or "Qwen2.5-7B-Instruct").strip()
        model = st.text_input(
            "模型名称",
            value=model_default,
            key=f"model_pref_model_{module_key}",
            help="不同厂商的模型命名不同；此处填模型字符串即可。",
        )

        # temperature
        temp_val = current.get("temperature")
        try:
            temp_default = float(temp_val) if temp_val is not None else float(default_temp)
        except Exception:
            temp_default = float(default_temp)
        temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=1.0,
            value=max(0.0, min(1.0, temp_default)),
            step=0.05,
            key=f"model_pref_temp_{module_key}",
        )

        # provider-specific
        api_base = None
        if provider == "siliconflow":
            api_base_default = (current.get("api_base") or "https://api.siliconflow.cn/v1").strip()
            api_base = st.text_input(
                "API Base（可选）",
                value=api_base_default,
                key=f"model_pref_api_base_{module_key}",
                help="默认使用硅基流动官方 base。若你有代理/自建网关，可在此覆盖。",
            )
            if not settings.SILICONFLOW_API_KEY:
                st.warning("当前未配置 `SILICONFLOW_API_KEY`，选择硅基流动模型将无法调用。")
        elif provider == "antigravity":
            api_base_default = (current.get("api_base") or getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip()
            api_base = st.text_input(
                "API Base（必填）",
                value=api_base_default,
                key=f"model_pref_api_base_{module_key}",
                help="从 Antigravity Tools 的“快速接入/Quick Integration”复制 base_url（通常以 /v1 或 /api/v1 结尾）。",
            )
            if not getattr(settings, "ANTIGRAVITY_API_KEY", ""):
                st.warning("当前未配置 `ANTIGRAVITY_API_KEY`，选择 Antigravity 将无法调用。")
            if not (api_base or "").strip() and not (getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip():
                st.warning("当前未配置 `ANTIGRAVITY_API_BASE`，并且该模块未填写 API Base，将无法调用。")
        else:
            if not settings.GOOGLE_API_KEY:
                st.warning("当前未配置 `GOOGLE_API_KEY`，选择 Gemini 模型将无法调用。")

        # 常见误配置修正：Qwen 不是 Gemini 模型，若选择了 Google provider 会导致 404。
        save_provider = provider
        save_api_base = api_base if provider in {"siliconflow", "antigravity"} else None
        openai_compat_providers = {"siliconflow", "antigravity"}
        if "qwen" in (model or "").lower() and provider not in openai_compat_providers:
            prefer_antigravity = bool(getattr(settings, "ANTIGRAVITY_API_KEY", "")) and bool((getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip())
            if prefer_antigravity:
                st.info("检测到 **Qwen** 模型：保存时将自动切换为 **Antigravity（OpenAI兼容）**（避免 Google/Gemini 侧 404）。")
                save_provider = "antigravity"
                save_api_base = (api_base or getattr(settings, "ANTIGRAVITY_API_BASE", "") or "").strip() or None
            else:
                st.info("检测到 **Qwen** 模型：保存时将自动切换为 **硅基流动**（避免 Google/Gemini 侧 404）。")
                save_provider = "siliconflow"
                save_api_base = api_base or "https://api.siliconflow.cn/v1"

        col_a, col_b = st.columns([1, 3])
        with col_a:
            if st.button("保存", use_container_width=True, key=f"model_pref_save_{module_key}"):
                st.session_state.db_manager.upsert_user_model_preference(
                    user_id=user_id,
                    module=module_key,
                    provider=save_provider,
                    model=model,
                    api_base=save_api_base,
                    temperature=float(temperature),
                )
                st.success("已保存")
                st.rerun()

        st.markdown("---")


def data_management_section():
    """数据管理部分"""
    st.markdown("### 🗄️ 数据管理")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 数据统计")
        docs = st.session_state.db_manager.get_user_documents(st.session_state.user["id"])
        
        st.markdown(f"- 文档数量: {len(docs)}")
        total_size = sum(d.file_size for d in docs) / 1024 / 1024
        st.markdown(f"- 总大小: {total_size:.2f} MB")
        total_chunks = sum(d.chunk_count for d in docs)
        st.markdown(f"- 知识片段: {total_chunks}")
    
    with col2:
        st.markdown("#### 危险操作")
        st.warning("以下操作不可恢复，请谨慎操作")
        
        if st.button("🗑️ 清空所有文档", type="secondary"):
            st.session_state.confirm_delete = True
        
        if st.session_state.get("confirm_delete"):
            st.error("确定要删除所有文档吗？此操作不可恢复！")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("确认删除", type="primary"):
                    # 删除所有文档
                    for doc in docs:
                        st.session_state.db_manager.delete_document(doc.id)
                    st.session_state.confirm_delete = False
                    st.success("已删除所有文档")
                    st.rerun()
            with col_b:
                if st.button("取消"):
                    st.session_state.confirm_delete = False
                    st.rerun()


def main():
    """主函数"""
    if not init_session():
        return
    
    sidebar()
    
    st.title("⚙️ 设置")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["API配置", "用户设置", "RAG参数", "数据管理", "模型选择"])
    
    with tab1:
        api_settings_section()
    
    with tab2:
        user_settings_section()
    
    with tab3:
        rag_settings_section()
    
    with tab4:
        data_management_section()

    with tab5:
        model_preferences_section()


if __name__ == "__main__":
    main()
