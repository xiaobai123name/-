"""
文档管理页面
支持上传、解析、管理学习文档
"""

import streamlit as st
from pathlib import Path
import sys
import os
import shutil
from datetime import datetime

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.database.crud import DatabaseManager
from backend.document.parser import DocumentParser
from backend.document.chunker import SmartChunker
from backend.document.embedder import DocumentEmbedder
from backend.retrieval.vector_store import VectorStore
from app.auth_cookie import ensure_auth_state_defaults, restore_user_from_cookie


st.set_page_config(
    page_title="文档管理 - 学习伴侣",
    page_icon="📁",
    layout="wide"
)


def init_session():
    """初始化会话"""
    ensure_auth_state_defaults()

    # 先初始化 db_manager
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
    
    # db_manager 已初始化，跳过
    if "db_manager" not in st.session_state:
        settings.ensure_directories()
        st.session_state.db_manager = DatabaseManager(str(settings.database_path))
    
    if "document_parser" not in st.session_state:
        st.session_state.document_parser = DocumentParser(
            use_llama_parse=bool(settings.LLAMA_CLOUD_API_KEY),
            llama_api_key=settings.LLAMA_CLOUD_API_KEY
        )
    
    if "chunker" not in st.session_state:
        st.session_state.chunker = SmartChunker()
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    return True


def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.markdown("### 📁 文档管理")
        
        if st.button("← 返回主页", use_container_width=True):
            st.switch_page("主页.py")
        
        st.markdown("---")
        
        # 文档统计
        docs = st.session_state.db_manager.get_user_documents(st.session_state.user["id"])
        completed = len([d for d in docs if d.process_status == "completed"])
        processing = len([d for d in docs if d.process_status == "processing"])
        
        st.markdown("#### 📊 文档统计")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("已处理", completed)
        with col2:
            st.metric("处理中", processing)
        
        st.markdown("---")
        
        # 支持的格式
        st.markdown("#### 📎 支持格式")
        st.markdown("- PDF 文档")
        st.markdown("- Word 文档 (.docx)")
        st.markdown("- Markdown 文件 (.md)")


def process_document(uploaded_file, user_id: str, progress_bar, status_text):
    """处理上传的文档"""
    try:
        # 保存文件
        status_text.text("正在保存文件...")
        progress_bar.progress(10)
        
        upload_dir = settings.upload_path / user_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 创建数据库记录
        status_text.text("正在创建记录...")
        progress_bar.progress(20)
        
        file_type = Path(uploaded_file.name).suffix.lstrip(".")
        doc = st.session_state.db_manager.create_document(
            user_id=user_id,
            filename=uploaded_file.name,
            file_type=file_type,
            file_path=str(file_path),
            file_size=uploaded_file.size
        )
        
        # 解析文档
        status_text.text("正在解析文档内容...")
        progress_bar.progress(30)
        
        st.session_state.db_manager.update_document_status(doc.id, "processing")
        parsed = st.session_state.document_parser.parse(str(file_path))
        
        # 分块
        status_text.text("正在智能分块...")
        progress_bar.progress(50)
        
        parent_chunks, child_chunks = st.session_state.chunker.chunk_document(parsed, doc.id)
        
        # 保存分块到数据库
        status_text.text("正在保存分块...")
        progress_bar.progress(60)
        
        all_chunks = parent_chunks + child_chunks
        chunks_data = [st.session_state.chunker.chunk_to_dict(c) for c in all_chunks]
        st.session_state.db_manager.create_chunks(chunks_data)
        
        # 向量化并存储
        status_text.text("正在向量化（这可能需要一些时间）...")
        progress_bar.progress(70)
        
        # 只对子切片进行向量化
        st.session_state.vector_store.add_chunks(child_chunks, user_id)
        
        # 也存储父切片（用于上下文）
        status_text.text("正在存储父文档...")
        progress_bar.progress(90)
        
        st.session_state.vector_store.add_chunks(parent_chunks, user_id)
        
        # 更新状态
        st.session_state.db_manager.update_document_status(
            doc.id, "completed", len(child_chunks)
        )
        
        progress_bar.progress(100)
        status_text.text("处理完成！")
        
        return True, f"文档「{uploaded_file.name}」处理完成，共生成 {len(child_chunks)} 个知识片段"
        
    except Exception as e:
        st.session_state.db_manager.update_document_status(doc.id, "failed")
        return False, f"处理失败: {str(e)}"


def display_document_card(doc):
    """显示文档卡片"""
    status_colors = {
        "pending": "🟡",
        "processing": "🔵",
        "completed": "🟢",
        "failed": "🔴"
    }
    
    status_texts = {
        "pending": "等待处理",
        "processing": "处理中",
        "completed": "已完成",
        "failed": "处理失败"
    }
    
    with st.container():
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"""
            **📄 {doc.filename}**  
            {status_colors.get(doc.process_status, '⚪')} {status_texts.get(doc.process_status, '未知')}
            | 大小: {doc.file_size / 1024:.1f} KB
            | 片段: {doc.chunk_count}
            """)
        
        with col2:
            st.text(doc.upload_time.strftime("%Y-%m-%d") if doc.upload_time else "")
        
        with col3:
            if st.button("🗑️ 删除", key=f"del_{doc.id}"):
                # 删除向量存储中的数据
                st.session_state.vector_store.delete_document_chunks(
                    doc.id, st.session_state.user["id"]
                )
                # 删除数据库记录
                st.session_state.db_manager.delete_document(doc.id)
                # 删除文件
                if os.path.exists(doc.file_path):
                    os.remove(doc.file_path)
                st.rerun()


def main():
    """主函数"""
    if not init_session():
        return
    
    sidebar()
    
    st.title("📁 文档管理")
    st.markdown("上传您的学习资料，系统将自动解析并构建知识库")
    
    # 文件上传区
    st.markdown("### 📤 上传文档")
    
    uploaded_files = st.file_uploader(
        "拖拽或选择文件上传",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        help="支持 PDF、Word、Markdown 格式"
    )
    
    if uploaded_files:
        if st.button("开始处理", type="primary"):
            for uploaded_file in uploaded_files:
                st.markdown(f"**处理文件:** {uploaded_file.name}")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success, message = process_document(
                    uploaded_file,
                    st.session_state.user["id"],
                    progress_bar,
                    status_text
                )
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
            
            st.rerun()
    
    st.markdown("---")
    
    # 文档列表
    st.markdown("### 📚 我的文档")
    
    docs = st.session_state.db_manager.get_user_documents(st.session_state.user["id"])
    
    if docs:
        for doc in docs:
            display_document_card(doc)
            st.markdown("---")
    else:
        st.info("暂无文档，请上传您的学习资料")


if __name__ == "__main__":
    main()
