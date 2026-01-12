"""
知识图谱页面
交互式可视化知识结构，支持实体探索和跨文档关联
"""

import streamlit as st
from pathlib import Path
import sys
import re
from typing import Optional

ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.config import settings
from backend.database.crud import DatabaseManager
from backend.retrieval.vector_store import VectorStore
from backend.learning.kg_builder import KnowledgeGraphBuilder, KnowledgeGraph, EntityType

# 导入可视化组件
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGRAPH_AVAILABLE = True
except ImportError:
    AGRAPH_AVAILABLE = False

# 导入 cookie 管理器
try:
    import extra_streamlit_components as stx
    COOKIE_MANAGER_AVAILABLE = True
    COOKIE_KEY = "app_user_id"
    COOKIE_WIDGET_KEY = "app_cookie_manager"
    _cookie_manager_kg = None
except ImportError:
    COOKIE_MANAGER_AVAILABLE = False


def get_cookie_manager():
    """获取 Cookie 管理器实例"""
    if not COOKIE_MANAGER_AVAILABLE:
        return None
    global _cookie_manager_kg
    if _cookie_manager_kg is None:
        _cookie_manager_kg = stx.CookieManager(key=COOKIE_WIDGET_KEY)
    return _cookie_manager_kg


st.set_page_config(
    page_title="知识图谱 - 学习伴侣",
    page_icon="🕸️",
    layout="wide"
)


# 自定义CSS
st.markdown("""
<style>
.entity-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
    color: white;
}
.entity-card h4 {
    margin: 0 0 8px 0;
    color: white;
}
.entity-card p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.9em;
}
.stat-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    border: 1px solid #e9ecef;
}
.stat-number {
    font-size: 24px;
    font-weight: bold;
    color: #2c3e50;
}
.stat-label {
    font-size: 12px;
    color: #6c757d;
}
.type-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


def init_session():
    """初始化会话"""
    if "db_manager" not in st.session_state:
        settings.ensure_directories()
        st.session_state.db_manager = DatabaseManager(str(settings.database_path))
    
    # 尝试从 cookie 恢复登录状态
    if ("user" not in st.session_state or st.session_state.user is None) and COOKIE_MANAGER_AVAILABLE:
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
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStore()
    
    if "kg_builder" not in st.session_state:
        st.session_state.kg_builder = KnowledgeGraphBuilder(
            vector_store=st.session_state.vector_store,
            db_manager=st.session_state.db_manager
        )
    
    if "current_graph" not in st.session_state:
        st.session_state.current_graph = None
    
    if "selected_entity" not in st.session_state:
        st.session_state.selected_entity = None
    
    return True


def sidebar():
    """侧边栏"""
    with st.sidebar:
        st.markdown("### 🕸️ 知识图谱")
        
        # 返回主页
        if st.button("← 返回主页", use_container_width=True):
            st.switch_page("主页.py")
        
        st.markdown("---")
        
        # 文档选择
        st.markdown("#### 📚 选择文档")
        docs = st.session_state.db_manager.get_user_documents(
            st.session_state.user["id"],
            status="completed"
        )
        
        if docs:
            doc_options = {"全部文档": None}
            doc_options.update({doc.filename: doc.id for doc in docs})
            
            selected_doc_name = st.selectbox(
                "选择要可视化的文档",
                options=list(doc_options.keys()),
                label_visibility="collapsed"
            )
            
            st.session_state.selected_doc_for_kg = doc_options[selected_doc_name]
            
            # 构建图谱按钮
            if st.button("🔨 构建知识图谱", use_container_width=True, type="primary"):
                with st.spinner("正在抽取知识实体和关系..."):
                    try:
                        if st.session_state.selected_doc_for_kg:
                            # 单文档
                            graph = st.session_state.kg_builder.build_from_document(
                                document_id=st.session_state.selected_doc_for_kg,
                                user_id=st.session_state.user["id"]
                            )
                        else:
                            # 全部文档
                            doc_ids = [d.id for d in docs]
                            graph = st.session_state.kg_builder.build_from_documents(
                                document_ids=doc_ids,
                                user_id=st.session_state.user["id"],
                                align_entities=True
                            )
                        
                        st.session_state.current_graph = graph
                        st.success(f"✅ 构建完成！发现 {len(graph.entities)} 个实体，{len(graph.relations)} 条关系")
                    except Exception as e:
                        st.error(f"构建失败: {str(e)}")
        else:
            st.info("暂无文档，请先上传")
        
        st.markdown("---")
        
        # 图谱统计
        if st.session_state.current_graph:
            graph = st.session_state.current_graph
            st.markdown("#### 📊 图谱统计")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("实体数", len(graph.entities))
            with col2:
                st.metric("关系数", len(graph.relations))
            
            # 实体类型分布
            type_counts = {}
            for entity in graph.entities.values():
                t = entity.type.value
                type_counts[t] = type_counts.get(t, 0) + 1
            
            st.markdown("**实体类型分布**")
            for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                st.caption(f"• {t}: {count}")
        
        st.markdown("---")
        
        # 搜索实体
        st.markdown("#### 🔍 搜索实体")
        search_query = st.text_input("输入关键词", placeholder="例如：神经网络")
        
        if search_query and st.session_state.current_graph:
            results = st.session_state.kg_builder.search_entities(
                query=search_query,
                graph=st.session_state.current_graph,
                top_k=5
            )
            
            if results:
                st.markdown("**搜索结果：**")
                for entity in results:
                    if st.button(f"📌 {entity.name}", key=f"search_{entity.id}"):
                        st.session_state.selected_entity = entity.id

        st.markdown("---")
        # 图谱显示设置（减少混杂 + 提升公式可读性）
        with st.expander("🎛️ 图谱显示设置", expanded=False):
            st.selectbox(
                "布局模式",
                options=["力导向（更分散）", "层级（更清晰）"],
                index=0,
                key="kg_layout_mode",
            )

            st.checkbox("聚焦模式（仅显示选中节点邻域）", value=True, key="kg_focus_mode")
            st.slider("邻域深度", min_value=1, max_value=3, value=1, key="kg_neighbor_depth")

            st.checkbox("简化数学公式/LaTeX（用于节点标签）", value=True, key="kg_humanize_math")
            st.checkbox("显示关系标签（会更拥挤）", value=False, key="kg_show_edge_labels")

            st.slider("节点标签最大长度", min_value=12, max_value=80, value=40, key="kg_max_label_len")
            st.slider("标签换行宽度（字符）", min_value=10, max_value=30, value=18, key="kg_wrap_width")

            st.slider("画布高度", min_value=450, max_value=950, value=650, key="kg_canvas_height")
            st.slider("画布宽度", min_value=800, max_value=1800, value=1200, key="kg_canvas_width")
            st.slider("节点间距（力导向）", min_value=120, max_value=420, value=220, key="kg_node_distance")


def get_type_color(entity_type: EntityType) -> str:
    """获取实体类型对应的颜色"""
    colors = {
        EntityType.CONCEPT: "#4ECDC4",
        EntityType.FORMULA: "#FF6B6B",
        EntityType.THEOREM: "#45B7D1",
        EntityType.EXAMPLE: "#96CEB4",
        EntityType.PERSON: "#DDA0DD",
        EntityType.METHOD: "#FFD93D",
        EntityType.APPLICATION: "#98D8C8",
    }
    return colors.get(entity_type, "#888888")


def get_type_label(entity_type: EntityType) -> str:
    """获取实体类型的中文标签"""
    labels = {
        EntityType.CONCEPT: "概念",
        EntityType.FORMULA: "公式",
        EntityType.THEOREM: "定理",
        EntityType.EXAMPLE: "示例",
        EntityType.PERSON: "人物",
        EntityType.METHOD: "方法",
        EntityType.APPLICATION: "应用",
    }
    return labels.get(entity_type, "其他")


# ==================== 文本可读化（公式/LaTeX） ====================

_LATEX_REPL = {
    r"\cdot": "·",
    r"\times": "×",
    r"\div": "÷",
    r"\neq": "≠",
    r"\leq": "≤",
    r"\geq": "≥",
    r"\pm": "±",
    r"\approx": "≈",
    r"\infty": "∞",
    r"\rightarrow": "→",
    r"\leftarrow": "←",
}


def humanize_math(text: str) -> str:
    """将常见 LaTeX/公式写法转换为更易读的文本（不追求完全等价渲染）。"""
    if not text:
        return ""
    s = (text or "").strip()

    # 去掉 $...$ / $$...$$
    s = re.sub(r"^\${1,2}", "", s)
    s = re.sub(r"\${1,2}$", "", s)

    # \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

    # ^{...}, _{...} 去花括号
    s = re.sub(r"\^\{([^{}]+)\}", r"^\1", s)
    s = re.sub(r"_\{([^{}]+)\}", r"_\1", s)

    # 常见符号替换
    for k, v in _LATEX_REPL.items():
        s = s.replace(k, v)

    # 清理残留花括号/多空格
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wrap_label(text: str, width: int = 18) -> str:
    """长标签换行，减少画布重叠。"""
    s = text or ""
    if width <= 0:
        return s
    return "\n".join(s[i : i + width] for i in range(0, len(s), width))


def truncate_text(text: str, max_len: int = 48) -> str:
    s = text or ""
    if max_len <= 0:
        return ""
    if len(s) <= max_len:
        return s
    return s[: max(1, max_len - 1)] + "…"


def extract_inline_latex(text: str) -> Optional[str]:
    """提取 $...$ 中的 LaTeX（用于 st.latex），提取不到则返回 None。"""
    if not text:
        return None
    m = re.search(r"\$(.+?)\$", text)
    if m:
        return m.group(1).strip()
    return None


def get_shape(entity_type: EntityType) -> str:
    """不同类型用不同形状，减少视觉混杂。"""
    shapes = {
        EntityType.CONCEPT: "dot",
        EntityType.FORMULA: "diamond",
        EntityType.THEOREM: "star",
        EntityType.EXAMPLE: "square",
        EntityType.PERSON: "triangle",
        EntityType.METHOD: "hexagon",
        EntityType.APPLICATION: "triangleDown",
    }
    return shapes.get(entity_type, "dot")


def render_graph(graph: KnowledgeGraph):
    """渲染知识图谱"""
    if not AGRAPH_AVAILABLE:
        st.warning("⚠️ 图谱可视化组件未安装，请运行: `pip install streamlit-agraph`")
        
        # 降级显示：文本列表
        st.markdown("### 📋 实体列表")
        for entity in graph.entities.values():
            st.markdown(f"**{entity.name}** ({get_type_label(entity.type)})")
            st.caption(entity.description)
        
        st.markdown("### 🔗 关系列表")
        for relation in graph.relations:
            source = graph.entities.get(relation.source_id)
            target = graph.entities.get(relation.target_id)
            if source and target:
                st.markdown(f"• {source.name} → {relation.type.value.replace('_', ' ')} → {target.name}")
        return
    
    # ===== 读取显示配置（侧边栏写入 session_state）=====
    layout_mode = st.session_state.get("kg_layout_mode", "力导向（更分散）")
    focus_mode = bool(st.session_state.get("kg_focus_mode", True))
    neighbor_depth = int(st.session_state.get("kg_neighbor_depth", 1))
    humanize = bool(st.session_state.get("kg_humanize_math", True))
    wrap_width = int(st.session_state.get("kg_wrap_width", 18))
    max_label_len = int(st.session_state.get("kg_max_label_len", 40))
    show_edge_labels = bool(st.session_state.get("kg_show_edge_labels", False))
    canvas_height = int(st.session_state.get("kg_canvas_height", 650))
    canvas_width = int(st.session_state.get("kg_canvas_width", 1200))
    node_distance = int(st.session_state.get("kg_node_distance", 220))

    # 聚焦模式：选中节点时仅显示其邻域子图
    if focus_mode and st.session_state.get("selected_entity"):
        selected_id = st.session_state.selected_entity
        if selected_id in graph.entities:
            try:
                graph = st.session_state.kg_builder.get_entity_neighbors(
                    entity_id=selected_id,
                    graph=graph,
                    depth=max(1, neighbor_depth),
                )
            except Exception:
                pass

    # 构建节点
    nodes = []
    for entity_id, entity in graph.entities.items():
        is_selected = entity_id == st.session_state.selected_entity
        label_text = entity.name
        if humanize:
            label_text = humanize_math(label_text)
        label_text = truncate_text(label_text, max_label_len)
        label_text = wrap_label(label_text, wrap_width)
        nodes.append(Node(
            id=entity_id,
            label=label_text,
            size=35 if is_selected else 25,
            color=get_type_color(entity.type),
            shape=get_shape(entity.type),
            # 避免 streamlit-agraph 双击时把 title 当 URL 打开造成错误
            title="about:blank",
            font={"color": "#2c3e50", "size": 14}
        ))
    
    # 构建边
    edges = []
    relation_colors = {
        "prerequisite": "#e74c3c",
        "leads_to": "#3498db",
        "belongs_to": "#9b59b6",
        "example_of": "#2ecc71",
        "similar_to": "#f1c40f",
        "contains": "#1abc9c",
        "applies_to": "#e67e22",
        "derived_from": "#34495e",
    }
    
    for relation in graph.relations:
        edges.append(Edge(
            source=relation.source_id,
            target=relation.target_id,
            label=(relation.type.value.replace("_", " ") if show_edge_labels else ""),
            color=relation_colors.get(relation.type.value, "#888888"),
            type="CURVE_SMOOTH"
        ))
    
    # 配置
    if str(layout_mode).startswith("层级"):
        config = Config(
            width=canvas_width,
            height=canvas_height,
            directed=True,
            physics=False,
            hierarchical=True,
            direction="UD",
            levelSeparation=220,
            nodeSpacing=180,
            treeSpacing=260,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
            node={"labelProperty": "label", "renderLabel": True},
            link={"labelProperty": "label", "renderLabel": show_edge_labels},
        )
    else:
        config = Config(
            width=canvas_width,
            height=canvas_height,
            directed=True,
            physics=True,
            hierarchical=False,
            solver="repulsion",
            minVelocity=2,
            maxVelocity=50,
            stabilization=True,
            fit=True,
            timestep=0.5,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
            node={"labelProperty": "label", "renderLabel": True},
            link={"labelProperty": "label", "renderLabel": show_edge_labels},
        )

        # 额外调参：提高节点间距，减少“混在一起”
        try:
            config.physics["repulsion"] = {
                "nodeDistance": int(node_distance),
                "springLength": int(node_distance),
                "springConstant": 0.05,
            }
        except Exception:
            pass
    
    # 渲染
    selected_node = agraph(nodes=nodes, edges=edges, config=config)
    
    if selected_node:
        st.session_state.selected_entity = selected_node


def render_entity_detail(graph: KnowledgeGraph, entity_id: str):
    """渲染实体详情"""
    if entity_id not in graph.entities:
        return
    
    entity = graph.entities[entity_id]
    
    st.markdown(f"""
    <div class="entity-card">
        <h4>🏷️ {entity.name}</h4>
        <p>{entity.description}</p>
    </div>
    """, unsafe_allow_html=True)

    # 公式/LaTeX：给出更适合人类阅读的展示
    if entity.type == EntityType.FORMULA:
        st.markdown("---")
        st.markdown("#### 🧮 公式展示")

        readable = humanize_math(entity.name)
        if readable:
            st.markdown("**可读版本**")
            st.code(readable)

        latex_expr = extract_inline_latex(entity.name)
        if latex_expr:
            st.markdown("**LaTeX 渲染**")
            try:
                st.latex(latex_expr)
            except Exception:
                pass
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**类型**")
        color = get_type_color(entity.type)
        st.markdown(f"""
        <span class="type-badge" style="background: {color}; color: white;">
            {get_type_label(entity.type)}
        </span>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("**来源文档**")
        for doc_id in entity.document_ids[:3]:
            doc = st.session_state.db_manager.get_document_by_id(doc_id)
            if doc:
                st.caption(f"📄 {doc.filename}")
    
    # 显示关联实体
    st.markdown("---")
    st.markdown("#### 🔗 关联实体")
    
    related = []
    for relation in graph.relations:
        if relation.source_id == entity_id:
            target = graph.entities.get(relation.target_id)
            if target:
                related.append({
                    "name": target.name,
                    "type": get_type_label(target.type),
                    "relation": relation.type.value.replace("_", " "),
                    "direction": "→",
                    "id": target.id
                })
        elif relation.target_id == entity_id:
            source = graph.entities.get(relation.source_id)
            if source:
                related.append({
                    "name": source.name,
                    "type": get_type_label(source.type),
                    "relation": relation.type.value.replace("_", " "),
                    "direction": "←",
                    "id": source.id
                })
    
    if related:
        for item in related:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{item['direction']}** {item['name']} ({item['type']})")
                st.caption(f"关系: {item['relation']}")
            with col2:
                if st.button("查看", key=f"view_{item['id']}"):
                    st.session_state.selected_entity = item["id"]
                    st.rerun()
    else:
        st.caption("暂无关联实体")


def main():
    """主函数"""
    if not init_session():
        return
    
    sidebar()
    
    st.title("🕸️ 知识图谱")
    st.markdown("可视化探索文档中的知识结构，发现概念之间的关联")
    
    # 检查是否有文档
    docs = st.session_state.db_manager.get_user_documents(
        st.session_state.user["id"],
        status="completed"
    )
    
    if not docs:
        st.warning("⚠️ 您还没有上传任何文档，请先上传学习资料。")
        if st.button("前往上传文档"):
            st.switch_page("pages/2_文档管理.py")
        return
    
    # 检查是否已构建图谱
    if st.session_state.current_graph is None:
        st.info("👈 请在左侧选择文档并点击「构建知识图谱」开始")
        
        # 显示功能说明
        st.markdown("---")
        st.markdown("### ✨ 功能特性")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 实体抽取**
            
            自动识别文档中的核心概念、公式、定理、方法等知识实体
            """)
        
        with col2:
            st.markdown("""
            **🔗 关系发现**
            
            分析实体间的语义关系：前置知识、包含、派生、应用等
            """)
        
        with col3:
            st.markdown("""
            **🎨 交互可视化**
            
            点击节点查看详情，拖拽探索，搜索定位感兴趣的知识点
            """)
        
        st.markdown("---")
        
        # 图例
        st.markdown("### 🎨 实体类型图例")
        type_cols = st.columns(7)
        type_info = [
            (EntityType.CONCEPT, "概念"),
            (EntityType.FORMULA, "公式"),
            (EntityType.THEOREM, "定理"),
            (EntityType.EXAMPLE, "示例"),
            (EntityType.PERSON, "人物"),
            (EntityType.METHOD, "方法"),
            (EntityType.APPLICATION, "应用"),
        ]
        
        for col, (t, label) in zip(type_cols, type_info):
            with col:
                color = get_type_color(t)
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="width: 24px; height: 24px; border-radius: 50%; 
                         background: {color}; margin: 0 auto;"></div>
                    <small>{label}</small>
                </div>
                """, unsafe_allow_html=True)
        
        return
    
    # 显示知识图谱
    graph = st.session_state.current_graph
    
    # 主区域和详情面板
    if st.session_state.selected_entity:
        col_graph, col_detail = st.columns([2, 1])
        
        with col_graph:
            st.markdown("### 📊 知识网络")
            render_graph(graph)
        
        with col_detail:
            st.markdown("### 📌 实体详情")
            render_entity_detail(graph, st.session_state.selected_entity)
            
            if st.button("✕ 关闭详情", use_container_width=True):
                st.session_state.selected_entity = None
                st.rerun()
    else:
        st.markdown("### 📊 知识网络")
        st.caption("💡 点击节点查看详情，拖拽可调整布局")
        render_graph(graph)
    
    # 底部统计
    st.markdown("---")
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(graph.entities)}</div>
            <div class="stat-label">知识实体</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[1]:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{len(graph.relations)}</div>
            <div class="stat-label">关系连接</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[2]:
        doc_count = len(set(d for e in graph.entities.values() for d in e.document_ids))
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{doc_count}</div>
            <div class="stat-label">关联文档</div>
        </div>
        """, unsafe_allow_html=True)
    
    with stat_cols[3]:
        # 计算平均连接度
        if graph.entities:
            total_connections = len(graph.relations) * 2
            avg_degree = total_connections / len(graph.entities)
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-number">{avg_degree:.1f}</div>
                <div class="stat-label">平均连接度</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
