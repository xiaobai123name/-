"""
知识图谱构建器
从文档中自动抽取实体和关系，构建可视化知识图谱
"""

import json
import re
import hashlib
import asyncio
import time
import threading
import difflib
import unicodedata
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import settings
from ..retrieval.vector_store import VectorStore
from ..database.crud import DatabaseManager
from ..llm.router import ModelRouter

def _kg_fix_invalid_json_escapes(s: str) -> str:
    """
    Fix common invalid JSON escapes in LLM output.

    Typical case: LaTeX like \\alpha may be emitted as \\alpha (invalid JSON escape \\a).
    We convert invalid escapes like \\a -> \\\\a so that JSON becomes parseable.
    """
    if not s:
        return s

    out: List[str] = []
    i = 0
    n = len(s)

    def _is_hex4(t: str) -> bool:
        return bool(re.fullmatch(r"[0-9a-fA-F]{4}", t or ""))

    while i < n:
        ch = s[i]
        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        # backslash
        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue

        nxt = s[i + 1]

        # valid simple escapes
        if nxt in ['"', "\\", "/", "b", "f", "n", "r", "t"]:
            out.append("\\" + nxt)
            i += 2
            continue

        # unicode escape: \uXXXX
        if nxt == "u":
            if i + 6 <= n and _is_hex4(s[i + 2 : i + 6]):
                out.append("\\u" + s[i + 2 : i + 6])
                i += 6
                continue
            # invalid \u -> escape backslash
            out.append("\\\\u")
            i += 2
            continue

        # invalid escape -> escape the backslash
        out.append("\\\\" + nxt)
        i += 2

    return "".join(out)


class EntityType(Enum):
    """实体类型"""
    CONCEPT = "concept"           # 概念
    FORMULA = "formula"           # 公式
    THEOREM = "theorem"           # 定理
    EXAMPLE = "example"           # 示例
    PERSON = "person"             # 人物
    METHOD = "method"             # 方法
    APPLICATION = "application"   # 应用


class RelationType(Enum):
    """关系类型"""
    BELONGS_TO = "belongs_to"         # 属于
    PREREQUISITE = "prerequisite"     # 前置知识
    LEADS_TO = "leads_to"             # 导向
    EXAMPLE_OF = "example_of"         # 是...的例子
    APPLIES_TO = "applies_to"         # 应用于
    SIMILAR_TO = "similar_to"         # 相似于
    CONTAINS = "contains"             # 包含
    DERIVED_FROM = "derived_from"     # 派生自


@dataclass
class Entity:
    """知识实体"""
    id: str
    name: str
    type: EntityType
    description: str
    document_ids: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "document_ids": list(self.document_ids),
            "properties": dict(self.properties),
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        raw_type = data.get("type", EntityType.CONCEPT.value)
        try:
            etype = EntityType(raw_type)
        except Exception:
            etype = EntityType.CONCEPT

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            type=etype,
            description=data.get("description", "") or "",
            document_ids=data.get("document_ids") or [],
            properties=data.get("properties") or {},
            embedding=data.get("embedding"),
        )


@dataclass
class Relation:
    """实体关系"""
    id: str
    source_id: str
    target_id: str
    type: RelationType
    weight: float = 1.0
    description: Optional[str] = None
    document_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "weight": self.weight,
            "description": self.description,
            "document_ids": list(self.document_ids),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        raw_type = data.get("type", RelationType.LEADS_TO.value)
        try:
            rtype = RelationType(raw_type)
        except Exception:
            rtype = RelationType.LEADS_TO

        return cls(
            id=data.get("id", ""),
            source_id=data.get("source_id", ""),
            target_id=data.get("target_id", ""),
            type=rtype,
            weight=float(data.get("weight", 1.0) or 1.0),
            description=data.get("description"),
            document_ids=data.get("document_ids") or [],
        )


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: List[Relation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "relations": [r.to_dict() for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        graph = cls()

        entities_data = data.get("entities") or {}
        if isinstance(entities_data, list):
            for e in entities_data:
                entity = Entity.from_dict(e or {})
                if entity.id:
                    graph.entities[entity.id] = entity
        elif isinstance(entities_data, dict):
            for eid, e in entities_data.items():
                entity = Entity.from_dict(e or {})
                # 允许 key 做兜底
                if not entity.id:
                    entity.id = str(eid)
                graph.entities[entity.id] = entity

        for r in data.get("relations") or []:
            rel = Relation.from_dict(r or {})
            if rel.id and rel.source_id and rel.target_id:
                graph.relations.append(rel)

        return graph
    
    def to_vis_format(self) -> Dict[str, Any]:
        """转换为可视化格式（vis.js / pyvis）"""
        nodes = []
        edges = []
        
        # 实体类型到颜色的映射
        type_colors = {
            EntityType.CONCEPT: "#4ECDC4",
            EntityType.FORMULA: "#FF6B6B",
            EntityType.THEOREM: "#45B7D1",
            EntityType.EXAMPLE: "#96CEB4",
            EntityType.PERSON: "#DDA0DD",
            EntityType.METHOD: "#FFD93D",
            EntityType.APPLICATION: "#98D8C8",
        }
        
        type_shapes = {
            EntityType.CONCEPT: "dot",
            EntityType.FORMULA: "diamond",
            EntityType.THEOREM: "star",
            EntityType.EXAMPLE: "square",
            EntityType.PERSON: "triangle",
            EntityType.METHOD: "hexagon",
            EntityType.APPLICATION: "triangleDown",
        }
        
        for entity_id, entity in self.entities.items():
            nodes.append({
                "id": entity_id,
                "label": entity.name,
                "title": entity.description,
                "color": type_colors.get(entity.type, "#888888"),
                "shape": type_shapes.get(entity.type, "dot"),
                "size": 25,
                "type": entity.type.value,
                "properties": entity.properties
            })
        
        # 关系类型到样式的映射
        relation_styles = {
            RelationType.PREREQUISITE: {"color": "#e74c3c", "dashes": False},
            RelationType.LEADS_TO: {"color": "#3498db", "dashes": False},
            RelationType.BELONGS_TO: {"color": "#9b59b6", "dashes": True},
            RelationType.EXAMPLE_OF: {"color": "#2ecc71", "dashes": True},
            RelationType.SIMILAR_TO: {"color": "#f1c40f", "dashes": [5, 5]},
            RelationType.CONTAINS: {"color": "#1abc9c", "dashes": False},
            RelationType.APPLIES_TO: {"color": "#e67e22", "dashes": False},
            RelationType.DERIVED_FROM: {"color": "#34495e", "dashes": [10, 5]},
        }
        
        for relation in self.relations:
            style = relation_styles.get(
                relation.type, 
                {"color": "#888888", "dashes": False}
            )
            edges.append({
                "from": relation.source_id,
                "to": relation.target_id,
                "label": relation.type.value.replace("_", " "),
                "title": relation.description or "",
                "color": style["color"],
                "dashes": style["dashes"],
                "arrows": "to",
                "width": max(1, relation.weight * 2)
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def to_streamlit_agraph_format(self) -> Tuple[List, List]:
        """转换为 streamlit-agraph 格式"""
        from streamlit_agraph import Node, Edge
        
        type_colors = {
            EntityType.CONCEPT: "#4ECDC4",
            EntityType.FORMULA: "#FF6B6B",
            EntityType.THEOREM: "#45B7D1",
            EntityType.EXAMPLE: "#96CEB4",
            EntityType.PERSON: "#DDA0DD",
            EntityType.METHOD: "#FFD93D",
            EntityType.APPLICATION: "#98D8C8",
        }
        
        nodes = []
        edges = []
        
        for entity_id, entity in self.entities.items():
            nodes.append(Node(
                id=entity_id,
                label=entity.name,
                size=25,
                color=type_colors.get(entity.type, "#888888"),
                title=entity.description
            ))
        
        for relation in self.relations:
            edges.append(Edge(
                source=relation.source_id,
                target=relation.target_id,
                label=relation.type.value.replace("_", " "),
                type="CURVE_SMOOTH"
            ))
        
        return nodes, edges


@dataclass
class _KGAsyncContext:
    """单次构建任务的异步上下文（避免 asyncio 对象跨事件循环复用）"""

    semaphore: asyncio.Semaphore
    rate_lock: asyncio.Lock
    last_call_time: float = 0.0


class KnowledgeGraphBuilder:
    """知识图谱构建器"""

    CACHE_VERSION = 1
    
    EXTRACTION_PROMPT = """你是一个知识图谱构建专家。请从以下文本中抽取知识实体和它们之间的关系。

## 实体类型
- concept: 抽象概念、术语
- formula: 数学公式、方程
- theorem: 定理、定律
- example: 具体示例、案例
- person: 人物（科学家、学者等）
- method: 方法、算法、技术
- application: 实际应用

## 关系类型
- prerequisite: A是理解B的前置知识
- leads_to: A导向/引出B
- belongs_to: A属于B的范畴
- example_of: A是B的一个例子
- similar_to: A和B相似
- contains: A包含B
- applies_to: A应用于B
- derived_from: A从B派生

## 输入文本
{text}

## 输出格式（JSON）
```json
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "concept|formula|theorem|example|person|method|application",
      "description": "简短描述（一句话）"
    }}
  ],
  "relations": [
    {{
      "source": "源实体名称",
      "target": "目标实体名称",
      "type": "prerequisite|leads_to|belongs_to|example_of|similar_to|contains|applies_to|derived_from",
      "description": "关系的简短描述（可选）"
    }}
  ]
}}
```

请抽取关键的知识实体和关系，不要遗漏重要概念。用中文回复。"""

    # SiliconFlow/Qwen 更容易在输出中夹带 markdown 或出现未闭合 ``` / []，这里用更强约束的纯 JSON 输出格式。
    SILICONFLOW_EXTRACTION_PROMPT = """你是一个知识图谱构建专家。请从以下文本中抽取知识实体和它们之间的关系。

## 要求
- 只输出**严格合法的 JSON 对象**，不要输出任何解释、不要输出 markdown、不要输出代码块（不要包含 ```）。
- 必须包含两个字段：entities（数组）、relations（数组）。如果没有关系也必须给出空数组。
- 为避免输出过长：实体最多输出 25 个，关系最多输出 40 条。

## 实体类型
- concept: 抽象概念、术语
- formula: 数学公式、方程
- theorem: 定理、定律
- example: 具体示例、案例
- person: 人物（科学家、学者等）
- method: 方法、算法、技术
- application: 实际应用

## 关系类型
- prerequisite: A是理解B的前置知识
- leads_to: A导向/引出B
- belongs_to: A属于B的范畴
- example_of: A是B的一个例子
- similar_to: A和B相似
- contains: A包含B
- applies_to: A应用于B
- derived_from: A从B派生

## 输入文本
{text}

## 输出 JSON Schema（示例结构）
{{
  "entities": [
    {{"name": "实体名称", "type": "concept", "description": "一句话描述"}}
  ],
  "relations": [
    {{"source": "源实体名称", "target": "目标实体名称", "type": "leads_to", "description": "可选"}}
  ]
}}"""

    # SiliconFlow/Qwen 关系补全（当第一次输出缺失 relations 时进行第二次调用）
    SILICONFLOW_RELATION_PROMPT = """你是一个知识图谱关系抽取专家。

## 目标
基于输入文本与候选实体列表，抽取实体之间的关系。

## 强制要求
- 只输出**严格合法的 JSON 对象**，不要输出任何解释、不要输出 markdown、不要输出代码块（不要包含 ```）。
- 只允许输出下方 schema，不得新增字段。
- source/target 必须来自候选实体列表，并且名称必须完全一致。
- 如果确实抽不出关系，也必须输出空数组。
- 为避免输出过长：关系最多输出 60 条。

## 关系类型
- prerequisite: A是理解B的前置知识
- leads_to: A导向/引出B
- belongs_to: A属于B的范畴
- example_of: A是B的一个例子
- similar_to: A和B相似
- contains: A包含B
- applies_to: A应用于B
- derived_from: A从B派生

## 候选实体（名称列表）
{entity_list}

## 输入文本
{text}

## 输出 JSON Schema
{{
  "relations": [
    {{"source": "源实体名称", "target": "目标实体名称", "type": "leads_to", "description": "可选"}}
  ]
}}"""

    # 批量抽取（多文档/全局图）默认沿用单段抽取 prompt
    # 说明：输入文本可能包含多文档拼接内容，依然要求输出相同 JSON 结构。
    BATCH_EXTRACTION_PROMPT = EXTRACTION_PROMPT

    ALIGNMENT_PROMPT = """你是一个知识图谱对齐专家。请判断以下两个实体是否指向同一个概念。

实体A：
- 名称：{name_a}
- 类型：{type_a}
- 描述：{desc_a}

实体B：
- 名称：{name_b}
- 类型：{type_b}
- 描述：{desc_b}

如果它们是同一个概念的不同表述（比如"神经网络"和"Neural Network"，或"NN"和"神经网络"），回复 "SAME"。
如果它们是不同的概念，回复 "DIFFERENT"。

只回复 SAME 或 DIFFERENT，不要解释。"""

    def __init__(
        self,
        vector_store: VectorStore,
        db_manager: DatabaseManager,
        api_key: Optional[str] = None
    ):
        self.vector_store = vector_store
        self.db = db_manager
        # Google key 仍用于 embeddings（语义合并/实体对齐/实体搜索）
        self.api_key = api_key or settings.GOOGLE_API_KEY
        # 模型路由：按用户+模块选择不同 provider/model（仅影响 KG 的 LLM 抽取/对齐）
        self.model_router = ModelRouter(db_manager)

        # === 速率/并发配置（可通过 settings 扩展覆盖）===
        self.max_concurrency: int = int(getattr(settings, "KG_MAX_CONCURRENCY", 3))
        self.min_request_interval_sec: float = float(getattr(settings, "KG_MIN_REQUEST_INTERVAL_SEC", 4.0))

        # 抽取与合并策略
        self.extraction_max_chars: int = int(getattr(settings, "KG_EXTRACTION_MAX_CHARS", 8000))
        self.merge_target_chars: int = int(getattr(settings, "KG_MERGE_TARGET_CHARS", 4500))
        self.merge_max_chars: int = int(getattr(settings, "KG_MERGE_MAX_CHARS", min(self.extraction_max_chars, 7500)))
        self.semantic_merge_threshold: float = float(getattr(settings, "KG_SEMANTIC_MERGE_THRESHOLD", 0.75))

        # 实体对齐参数
        self.alignment_direct_threshold: float = float(getattr(settings, "KG_ALIGNMENT_DIRECT_THRESHOLD", 0.80))
        self.alignment_llm_threshold: float = float(getattr(settings, "KG_ALIGNMENT_LLM_THRESHOLD", 0.70))
        self.alignment_max_llm_checks: int = int(getattr(settings, "KG_ALIGNMENT_MAX_LLM_CHECKS", 50))

        # 是否启用持久化缓存（SQLite）
        self.enable_persistent_cache: bool = bool(getattr(settings, "KG_ENABLE_PERSISTENT_CACHE", True))

        # === 默认 LLM（无 user_id 场景兜底）===
        self._default_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=self.api_key,
            temperature=0.3,
            convert_system_message_to_human=True,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=self.api_key,
        )

        # 内存缓存：key -> KnowledgeGraph（单次进程内有效）
        self._graph_cache: Dict[str, KnowledgeGraph] = {}

    # ==================== LLM Router ====================

    def _get_llm(self, user_id: Optional[str] = None):
        """按用户偏好获取 KG 用的 LLM（异步调用会使用 .ainvoke）。"""
        if user_id:
            return self.model_router.get_chat_model(user_id=user_id, module="kg", streaming=False)
        return self._default_llm

    def _llm_fingerprint(self, user_id: Optional[str]) -> str:
        """将 provider/model/api_base 归一化为短指纹，用于缓存隔离。"""
        try:
            if user_id:
                cfg = self.model_router.get_module_llm_config(user_id=user_id, module="kg")
                raw = f"{cfg.provider}|{cfg.model}|{cfg.api_base or ''}"
            else:
                raw = f"google|{settings.LLM_MODEL}|"
        except Exception:
            raw = f"google|{settings.LLM_MODEL}|"

        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]

    # ==================== Cache Keys & Hashing ====================

    def _mem_cache_key(self, user_id: str, graph_key: str) -> str:
        return f"{user_id}|{graph_key}"

    def _graph_key_for_document(self, document_id: str, llm_fp: str) -> str:
        return f"kg:doc:v{self.CACHE_VERSION}:{document_id}:m:{llm_fp}"

    def _graph_key_for_documents(self, document_ids: List[str], align_entities: bool, llm_fp: str) -> str:
        # document_ids 可能很长，做一个短 fingerprint 避免 key 过长
        sorted_ids = sorted(set(document_ids))
        fingerprint = hashlib.md5("|".join(sorted_ids).encode("utf-8")).hexdigest()[:16]
        return f"kg:multi:v{self.CACHE_VERSION}:{fingerprint}:align:{int(bool(align_entities))}:m:{llm_fp}"

    def _hash_chunks(self, chunks: List[Any]) -> str:
        """对 chunks 内容做稳定哈希，用于缓存失效判断"""
        h = hashlib.sha256()
        for c in chunks:
            content = getattr(c, "content", "") or ""
            h.update(content.encode("utf-8", errors="ignore"))
            h.update(b"\0")
        return h.hexdigest()

    def _get_parent_chunks_for_kg(self, document_id: str) -> List[Any]:
        """知识图谱仅使用父 chunks（兜底：如果父 chunk 不存在则回退到全部 chunks）。"""
        chunks = self.db.get_document_chunks(document_id, chunk_type="parent")
        if not chunks:
            chunks = self.db.get_document_chunks(document_id)

        # 过滤太短内容
        return [c for c in chunks if (getattr(c, "content", "") or "").strip() and len(c.content) >= 50]

    # ==================== Async Runner ====================

    def _run_async(self, coro):
        """在同步上下文中安全运行协程（兼容已存在事件循环的环境）。"""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        # 如果当前线程已有事件循环（例如某些运行环境），使用新线程跑一个独立事件循环
        result_holder: Dict[str, Any] = {}
        error_holder: Dict[str, BaseException] = {}

        def _runner():
            try:
                result_holder["result"] = asyncio.run(coro)
            except BaseException as e:
                error_holder["error"] = e

        t = threading.Thread(target=_runner, daemon=True)
        t.start()
        t.join()
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result")

    def _new_async_context(self) -> _KGAsyncContext:
        """为单次构建创建异步上下文（避免跨 loop 复用 semaphore/lock）。"""
        return _KGAsyncContext(
            semaphore=asyncio.Semaphore(max(1, self.max_concurrency)),
            rate_lock=asyncio.Lock(),
            last_call_time=0.0,
        )

    # ==================== Persistence Cache (SQLite) ====================

    def _load_graph_from_persistent_cache(
        self,
        user_id: str,
        graph_key: str,
        source_hash: str,
    ) -> Optional[KnowledgeGraph]:
        if not self.enable_persistent_cache:
            return None

        record = self.db.get_kg_cache(user_id=user_id, graph_key=graph_key)
        if not record:
            return None
        if record.source_hash != source_hash:
            return None
        if not record.graph_data:
            return None

        try:
            graph = KnowledgeGraph.from_dict(record.graph_data)
        except Exception:
            return None

        # 保护性策略：空图谱很可能来自“抽取失败/解析失败”的错误缓存。
        # 仅当明确标记为 empty_chunks 时才认为空图谱可复用。
        if not graph.entities and not graph.relations:
            meta = getattr(record, "meta", None) or {}
            note = meta.get("note") if isinstance(meta, dict) else None
            if note != "empty_chunks":
                return None

        return graph

    def _save_graph_to_persistent_cache(
        self,
        user_id: str,
        graph_key: str,
        source_hash: str,
        graph: KnowledgeGraph,
        document_ids: List[str],
        scope: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enable_persistent_cache:
            return

        self.db.upsert_kg_cache(
            user_id=user_id,
            graph_key=graph_key,
            source_hash=source_hash,
            graph_data=graph.to_dict(),
            document_ids=document_ids,
            scope=scope,
            meta=meta,
        )

    def save_graph_snapshot(self, graph: KnowledgeGraph, snapshot_dir: Optional[str] = None) -> str:
        """
        将图谱保存为快照文件（pickle）。

        说明：
        - 快照用于调试/留档/对比，不参与缓存命中逻辑。
        - 为避免类结构变化导致旧快照不可读，快照保存 graph.to_dict()。
        """
        import pickle
        from datetime import datetime
        from pathlib import Path

        base_dir = Path(snapshot_dir) if snapshot_dir else (settings.BASE_DIR / "data" / "snapshots")
        base_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M")
        node_count = len(graph.entities)

        # 与现有命名保持一致：graph_YYYYMMDD_HHMM_{n}nodes.pkl
        path = base_dir / f"graph_{ts}_{node_count}nodes.pkl"
        if path.exists():
            # 同一分钟内多次保存时避免覆盖
            i = 1
            while True:
                candidate = base_dir / f"graph_{ts}_{node_count}nodes_{i}.pkl"
                if not candidate.exists():
                    path = candidate
                    break
                i += 1

        with open(path, "wb") as f:
            pickle.dump(graph.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

        return str(path)
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """生成实体ID"""
        raw = f"{name}_{entity_type}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    
    def _generate_relation_id(self, source: str, target: str, rel_type: str) -> str:
        """生成关系ID"""
        raw = f"{source}_{target}_{rel_type}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]
    
    def _parse_extraction_response(self, response: str) -> Dict:
        """解析抽取响应"""
        # 1) 优先提取 code-fence 内的 JSON（兼容 ```json 和 ```）
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response or "", flags=re.IGNORECASE)
        json_str = (json_match.group(1) if json_match else (response or "")).strip()

        # 兼容：只有开头 ```json 但没有闭合 ``` 的情况
        if not json_match and (json_str.lstrip().startswith("```")):
            nl = json_str.find("\n")
            if nl != -1:
                json_str = json_str[nl + 1 :].strip()
            # 如果末尾有孤立的 ``` 也一并剔除
            json_str = re.sub(r"\s*```+\s*$", "", json_str).strip()

        def _fix_trailing_commas(s: str) -> str:
            s = re.sub(r",\s*}", "}", s)
            s = re.sub(r",\s*]", "]", s)
            return s

        def _try_json_load(s: str) -> Optional[Any]:
            s = (s or "").strip()
            if not s:
                return None
            try:
                return json.loads(s)
            except Exception:
                return None

        def _try_raw_decode(s: str) -> Optional[Any]:
            s = (s or "")
            if not s:
                return None
            # 找到第一个可能的 JSON 起始符号
            candidates = [i for i in [s.find("{"), s.find("[")] if i >= 0]
            if not candidates:
                return None
            start = min(candidates)
            tail = s[start:].lstrip()
            try:
                decoder = json.JSONDecoder()
                obj, _end = decoder.raw_decode(tail)
                return obj
            except Exception:
                return None

        def _extract_objects_from_array_key(s: str, key: str) -> List[Dict[str, Any]]:
            """
            当整体 JSON 不可解析时的兜底：从 `"key": [ ... ]` 里逐个抽取 `{...}` 对象。
            该策略不依赖数组闭合 `]`，也不依赖对象之间的逗号分隔。
            """
            s = s or ""
            m = re.search(rf"\"{re.escape(key)}\"\s*:\s*\[", s, flags=re.IGNORECASE)
            if not m:
                return []
            i = m.end()  # 指向 '[' 后
            n = len(s)
            objs: List[str] = []
            obj_start: Optional[int] = None
            depth = 0
            in_str = False
            esc = False

            while i < n:
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == "\"":
                        in_str = False
                    i += 1
                    continue

                if ch == "\"":
                    in_str = True
                    i += 1
                    continue

                if obj_start is None:
                    if ch == "{":
                        obj_start = i
                        depth = 1
                    elif ch == "]":
                        break
                    i += 1
                    continue

                # within object
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        objs.append(s[obj_start : i + 1])
                        obj_start = None
                i += 1

            parsed: List[Dict[str, Any]] = []
            for t in objs:
                fixed = _fix_trailing_commas(_kg_fix_invalid_json_escapes(t))
                try:
                    obj = json.loads(fixed)
                    if isinstance(obj, dict):
                        parsed.append(obj)
                except Exception:
                    continue
            return parsed

        def _balance_json_closers(s: str) -> str:
            """
            修复常见“缺少闭合括号”的 LLM 输出：在不进入字符串的前提下，
            扫描 { [ } ] 的栈并补齐缺失的闭合符号。
            """
            s = (s or "")
            if not s:
                return s
            # 从第一个 { 或 [ 开始
            starts = [i for i in [s.find("{"), s.find("[")] if i >= 0]
            if not starts:
                return s
            s2 = s[min(starts) :]

            stack: List[str] = []
            in_str = False
            esc = False
            for ch in s2:
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == "\"":
                        in_str = False
                    continue

                if ch == "\"":
                    in_str = True
                    continue

                if ch in "{[":
                    stack.append(ch)
                elif ch == "}" and stack and stack[-1] == "{":
                    stack.pop()
                elif ch == "]" and stack and stack[-1] == "[":
                    stack.pop()
                else:
                    # unmatched closing or other char: ignore
                    pass

            closers = "".join("}" if t == "{" else "]" for t in reversed(stack))
            return s2 + closers

        parse_mode = "direct"
        result_obj: Optional[Any] = _try_json_load(json_str)
        if result_obj is None:
            # 2) 修复非法转义 + trailing comma 再试
            parse_mode = "fixed_escapes"
            fixed = _fix_trailing_commas(_kg_fix_invalid_json_escapes(json_str))
            result_obj = _try_json_load(fixed)

        if result_obj is None:
            # 3) 尝试从“包含 JSON 的文本”中 raw_decode（允许前后有解释文本）
            parse_mode = "raw_decode"
            result_obj = _try_raw_decode(json_str)

        if result_obj is None:
            parse_mode = "raw_decode_fixed"
            fixed = _fix_trailing_commas(_kg_fix_invalid_json_escapes(json_str))
            result_obj = _try_raw_decode(fixed)

        if result_obj is None:
            # 4) 尝试补齐缺失的闭合括号（常见于模型输出截断/漏写 ] 或 }）
            parse_mode = "balanced_fixed"
            fixed = _fix_trailing_commas(_kg_fix_invalid_json_escapes(json_str))
            balanced = _balance_json_closers(fixed)
            result_obj = _try_json_load(balanced)
            if result_obj is None:
                result_obj = _try_raw_decode(balanced)

        if not isinstance(result_obj, dict):
            # 最终兜底：尝试从 entities/relations 数组中逐个抽取对象
            ents_fb = _extract_objects_from_array_key(json_str, "entities")
            rels_fb = _extract_objects_from_array_key(json_str, "relations")
            result: Dict[str, Any] = {"entities": ents_fb, "relations": rels_fb}
        else:
            result = result_obj

        # 4) 结构归一化：entities/relations 必须是 List[dict]
        ents = result.get("entities", []) or []
        rels = result.get("relations", []) or []
        if not isinstance(ents, list):
            if isinstance(ents, dict) and ("name" in ents and "type" in ents):
                ents = [ents]
            else:
                ents = []
        if not isinstance(rels, list):
            if isinstance(rels, dict) and ("source" in rels and "target" in rels and "type" in rels):
                rels = [rels]
            else:
                rels = []
        result["entities"] = ents
        result["relations"] = rels

        return result

    # ==================== Embeddings (with retry) ====================

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _embed_documents_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """对单个 batch 做 embedding（带重试）。"""
        return self.embeddings.embed_documents(texts)

    def _embed_documents_with_retry(self, texts: List[str]) -> List[List[float]]:
        """对多段文本做 embedding（分批 + 单批重试）。"""
        if not texts:
            return []

        batch_size = int(getattr(settings, "KG_EMBED_BATCH_SIZE", 64))
        batch_size = max(1, batch_size)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed_documents_batch_with_retry(batch))
        return all_embeddings

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _embed_query_with_retry(self, query: str) -> List[float]:
        return self.embeddings.embed_query(query)

    # ==================== LLM Calls (async + semaphore + rate limit + retry) ====================

    async def _wait_rate_limit(self, ctx: _KGAsyncContext) -> None:
        """全局速率控制：保证任意两次 LLM 请求的启动间隔 >= min_request_interval_sec"""
        if self.min_request_interval_sec <= 0:
            return

        async with ctx.rate_lock:
            now = time.monotonic()
            elapsed = now - ctx.last_call_time
            wait_sec = self.min_request_interval_sec - elapsed
            if wait_sec > 0:
                await asyncio.sleep(wait_sec)
            ctx.last_call_time = time.monotonic()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _ainvoke_with_retry(
        self,
        messages: List[HumanMessage],
        ctx: _KGAsyncContext,
        user_id: Optional[str] = None,
    ):
        async with ctx.semaphore:
            await self._wait_rate_limit(ctx)
            llm = self._get_llm(user_id=user_id)
            return await llm.ainvoke(messages)

    # ==================== Chunk Merge Strategy ====================

    def _size_merge_texts(self, texts: List[str]) -> List[str]:
        """纯长度合并：当语义合并失败时作为兜底策略"""
        merged: List[str] = []
        buf: List[str] = []
        buf_len = 0
        sep = "\n\n---\n\n"

        for t in texts:
            t = (t or "").strip()
            if not t:
                continue

            t_len = len(t)
            # 单段过长：直接截断/单独处理
            if t_len >= self.merge_max_chars:
                if buf:
                    merged.append(sep.join(buf))
                    buf, buf_len = [], 0
                merged.append(t[: self.merge_max_chars])
                continue

            # 加入后超上限：先 flush
            if buf and (buf_len + len(sep) + t_len) > self.merge_max_chars:
                merged.append(sep.join(buf))
                buf, buf_len = [], 0

            buf.append(t)
            buf_len = buf_len + (len(sep) if buf_len > 0 else 0) + t_len

            # 达到目标长度就 flush，避免过小批次
            if buf_len >= self.merge_target_chars:
                merged.append(sep.join(buf))
                buf, buf_len = [], 0

        if buf:
            merged.append(sep.join(buf))

        return merged

    def _semantic_merge_chunks(self, chunks: List[Any]) -> List[str]:
        """
        语义感知分块合并（顺序合并相邻 chunk）：
        - 相似度高时优先合并
        - 同时约束合并后的长度不超过 merge_max_chars
        """
        texts: List[str] = [(getattr(c, "content", "") or "").strip() for c in chunks]
        texts = [t for t in texts if t]
        if len(texts) <= 1:
            return texts

        try:
            embeddings = self._embed_documents_with_retry(texts)
        except Exception:
            # embeddings 失败时回退到纯长度合并
            return self._size_merge_texts(texts)

        merged: List[str] = []
        sep = "\n\n---\n\n"

        cur_texts: List[str] = [texts[0]]
        cur_len = len(texts[0])
        cur_emb = embeddings[0]
        cur_count = 1

        for i in range(1, len(texts)):
            t = texts[i]
            emb = embeddings[i]

            sim = self._cosine_similarity(cur_emb, emb)
            would_len = cur_len + len(sep) + len(t)

            should_merge = (sim >= self.semantic_merge_threshold) and (would_len <= self.merge_max_chars)
            # 如果当前 batch 还很短，也允许低相似度拼接，减少调用次数
            if not should_merge and cur_len < int(self.merge_target_chars * 0.6) and would_len <= self.merge_max_chars:
                should_merge = True

            if should_merge:
                cur_texts.append(t)
                cur_len = would_len
                # 更新 batch embedding（简单均值）
                cur_emb = [(a * cur_count + b) / (cur_count + 1) for a, b in zip(cur_emb, emb)]
                cur_count += 1
                continue

            merged.append(sep.join(cur_texts))
            cur_texts = [t]
            cur_len = len(t)
            cur_emb = emb
            cur_count = 1

        if cur_texts:
            merged.append(sep.join(cur_texts))

        return merged
    
    async def _extract_from_text_async(
        self,
        text: str,
        document_id: str,
        user_id: str,
        ctx: _KGAsyncContext,
    ) -> Tuple[List[Entity], List[Relation]]:
        """异步抽取实体与关系（并发 + 限流 + 重试）。"""
        # SiliconFlow/Qwen 使用更强约束的纯 JSON prompt，减少不闭合 ``` / [] 导致的解析失败
        prompt_tpl = self.EXTRACTION_PROMPT
        is_siliconflow = False
        try:
            cfg = self.model_router.get_module_llm_config(user_id=user_id, module="kg")
            if (getattr(cfg, "provider", "") or "").strip().lower() == "siliconflow":
                is_siliconflow = True
                prompt_tpl = self.SILICONFLOW_EXTRACTION_PROMPT
        except Exception:
            pass

        prompt = prompt_tpl.format(text=(text or "")[: self.extraction_max_chars])
        messages = [HumanMessage(content=prompt)]

        response = await self._ainvoke_with_retry(messages, ctx, user_id=user_id)

        result = self._parse_extraction_response(getattr(response, "content", "") or "")

        # SiliconFlow/Qwen 常见情况：只输出 entities，缺失 relations。这里做一次关系补全二次调用。
        try:
            rels_raw = (result.get("relations") or []) if isinstance(result, dict) else []
            ents_raw = (result.get("entities") or []) if isinstance(result, dict) else []
        except Exception:
            rels_raw, ents_raw = [], []

        if is_siliconflow and (not rels_raw) and isinstance(ents_raw, list) and len(ents_raw) >= 2:
            # 取前 N 个实体名，避免 prompt 过长
            names: List[str] = []
            for e in ents_raw:
                if not isinstance(e, dict):
                    continue
                nm = (e.get("name") or "").strip()
                if nm and nm not in names:
                    names.append(nm)
                if len(names) >= 25:
                    break

            if len(names) >= 2:
                rel_prompt = self.SILICONFLOW_RELATION_PROMPT.format(
                    entity_list="\n".join([f"- {n}" for n in names]),
                    text=(text or "")[: self.extraction_max_chars],
                )
                rel_messages = [HumanMessage(content=rel_prompt)]
                try:
                    rel_resp = await self._ainvoke_with_retry(rel_messages, ctx, user_id=user_id)
                    rel_result = self._parse_extraction_response(getattr(rel_resp, "content", "") or "")
                    rels2 = (rel_result.get("relations") or []) if isinstance(rel_result, dict) else []
                    if isinstance(rels2, list) and rels2:
                        result["relations"] = rels2
                except Exception:
                    # 关系补全失败不影响主流程
                    pass

        entities: List[Entity] = []
        relations: List[Relation] = []
        entity_name_to_id: Dict[str, str] = {}

        # 处理实体
        for e in result.get("entities", []) or []:
            if not isinstance(e, dict):
                continue
            name = (e.get("name") or "").strip()
            if not name:
                continue

            raw_type = (e.get("type") or EntityType.CONCEPT.value).strip()
            try:
                entity_type = EntityType(raw_type)
            except Exception:
                entity_type = EntityType.CONCEPT
                raw_type = entity_type.value

            entity_id = self._generate_entity_id(name, raw_type)
            entity = Entity(
                id=entity_id,
                name=name,
                type=entity_type,
                description=(e.get("description") or "").strip(),
                document_ids=[document_id],
            )
            entities.append(entity)
            entity_name_to_id[name] = entity_id

        # 处理关系
        for r in result.get("relations", []) or []:
            if not isinstance(r, dict):
                continue
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            if not src or not tgt:
                continue

            source_id = entity_name_to_id.get(src)
            target_id = entity_name_to_id.get(tgt)
            if not source_id or not target_id:
                continue

            raw_type = (r.get("type") or RelationType.LEADS_TO.value).strip()
            try:
                rel_type = RelationType(raw_type)
            except Exception:
                rel_type = RelationType.LEADS_TO
                raw_type = rel_type.value

            relation = Relation(
                id=self._generate_relation_id(source_id, target_id, raw_type),
                source_id=source_id,
                target_id=target_id,
                type=rel_type,
                description=(r.get("description") or None),
                document_ids=[document_id],
            )
            relations.append(relation)

        return entities, relations

    def extract_from_text(
        self,
        text: str,
        document_id: str,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Entity], List[Relation]]:
        """同步接口：内部使用异步并发 + 限流执行。"""

        async def _runner():
            ctx = self._new_async_context()
            uid = user_id or ""
            return await self._extract_from_text_async(text=text, document_id=document_id, user_id=uid, ctx=ctx)

        return self._run_async(_runner())
    
    def build_from_document(
        self,
        document_id: str,
        user_id: str
    ) -> KnowledgeGraph:
        """同步构建单文档知识图谱（内部走异步并发）。"""
        return self._run_async(self.build_from_document_async(document_id=document_id, user_id=user_id))

    async def build_from_document_async(
        self,
        document_id: str,
        user_id: str,
        ctx: Optional[_KGAsyncContext] = None,
    ) -> KnowledgeGraph:
        """异步构建单文档知识图谱：父 chunks + 语义合并 + 并发抽取 + 持久化缓存。"""
        llm_fp = self._llm_fingerprint(user_id)
        graph_key = self._graph_key_for_document(document_id, llm_fp)
        mem_key = self._mem_cache_key(user_id, graph_key)

        # 1) 内存缓存
        if mem_key in self._graph_cache:
            cached_mem = self._graph_cache[mem_key]
            # 空图谱可能是错误缓存，不直接命中（让后续逻辑重建/重试）
            if cached_mem.entities or cached_mem.relations:
                return cached_mem

        # 2) 读取父 chunks 并计算 hash
        chunks = self._get_parent_chunks_for_kg(document_id)
        source_hash = self._hash_chunks(chunks)

        # 3) 持久化缓存
        cached = self._load_graph_from_persistent_cache(user_id=user_id, graph_key=graph_key, source_hash=source_hash)
        if cached is not None:
            self._graph_cache[mem_key] = cached

            # SiliconFlow/Qwen 常见：缓存里有实体但无关系。这里在 cache hit 情况下补全关系，并回写缓存。
            try:
                cfg = self.model_router.get_module_llm_config(user_id=user_id, module="kg")
                is_sf = (getattr(cfg, "provider", "") or "").strip().lower() == "siliconflow"
            except Exception:
                is_sf = False

            if is_sf and len(getattr(cached, "entities", {}) or {}) >= 2 and len(getattr(cached, "relations", []) or []) == 0:
                if ctx is None:
                    ctx = self._new_async_context()

                # 构建候选实体名称列表（限制长度，避免 prompt 过长）
                names: List[str] = []
                for e in (cached.entities or {}).values():
                    nm = (getattr(e, "name", "") or "").strip()
                    if nm and nm not in names:
                        names.append(nm)
                    if len(names) >= 25:
                        break

                if len(names) >= 2:
                    try:
                        # 用父 chunks 文本做关系补全（长度受 extraction_max_chars 限制）
                        text_for_rel = "\n\n".join([(getattr(c, "content", "") or "") for c in (chunks or [])])
                        text_for_rel = (text_for_rel or "")[: self.extraction_max_chars]

                        rel_prompt = self.SILICONFLOW_RELATION_PROMPT.format(
                            entity_list="\n".join([f"- {n}" for n in names]),
                            text=text_for_rel,
                        )
                        rel_messages = [HumanMessage(content=rel_prompt)]
                        rel_resp = await self._ainvoke_with_retry(rel_messages, ctx, user_id=user_id)
                        rel_result = self._parse_extraction_response(getattr(rel_resp, "content", "") or "")
                        rels2 = (rel_result.get("relations") or []) if isinstance(rel_result, dict) else []

                        if isinstance(rels2, list) and rels2:
                            name_to_id: Dict[str, str] = { (getattr(e, "name", "") or ""): eid for eid, e in (cached.entities or {}).items() }
                            existing_rel_ids: Set[str] = set([r.id for r in (cached.relations or []) if getattr(r, "id", None)])

                            added = 0
                            for r in rels2:
                                if not isinstance(r, dict):
                                    continue
                                src_name = (r.get("source") or "").strip()
                                tgt_name = (r.get("target") or "").strip()
                                if not src_name or not tgt_name:
                                    continue
                                if src_name not in name_to_id or tgt_name not in name_to_id:
                                    continue

                                raw_type = (r.get("type") or "").strip()
                                if not raw_type:
                                    continue
                                try:
                                    rel_type = RelationType(raw_type)
                                except Exception:
                                    continue

                                source_id = name_to_id[src_name]
                                target_id = name_to_id[tgt_name]
                                rel_id = self._generate_relation_id(source_id, target_id, rel_type.value)
                                if rel_id in existing_rel_ids:
                                    continue
                                existing_rel_ids.add(rel_id)
                                cached.relations.append(
                                    Relation(
                                        id=rel_id,
                                        source_id=source_id,
                                        target_id=target_id,
                                        type=rel_type,
                                        description=(r.get("description") or None),
                                        document_ids=[document_id],
                                    )
                                )
                                added += 1

                            # 回写缓存（即使没新增也写 meta，避免反复补全）
                            self._graph_cache[mem_key] = cached
                            self._save_graph_to_persistent_cache(
                                user_id=user_id,
                                graph_key=graph_key,
                                source_hash=source_hash,
                                graph=cached,
                                document_ids=[document_id],
                                scope="document",
                                meta={
                                    "cache_version": self.CACHE_VERSION,
                                    "parent_chunks": len(chunks),
                                    "merged_batches": 0,
                                    "relations_backfilled": True,
                                    "relations_added": int(added),
                                },
                            )
                    except Exception:
                        # 关系补全失败不影响主流程（沿用缓存结果）
                        pass

            return cached

        # 4) 构建：语义感知合并 + 并发抽取
        if ctx is None:
            ctx = self._new_async_context()

        merged_texts = self._semantic_merge_chunks(chunks)
        if not merged_texts:
            graph = KnowledgeGraph()
            self._graph_cache[mem_key] = graph
            self._save_graph_to_persistent_cache(
                user_id=user_id,
                graph_key=graph_key,
                source_hash=source_hash,
                graph=graph,
                document_ids=[document_id],
                scope="document",
                meta={"note": "empty_chunks"},
            )
            return graph

        tasks = [
            self._extract_from_text_async(text=t, document_id=document_id, user_id=user_id, ctx=ctx)
            for t in merged_texts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        graph = KnowledgeGraph()
        existing_rel_ids: Set[str] = set()

        err_count = 0
        first_err: Optional[str] = None
        for res in results:
            if isinstance(res, Exception):
                err_count += 1
                if first_err is None:
                    first_err = f"{type(res).__name__}: {(str(res) or '')[:200]}"
                continue
            entities, relations = res

            # 合并实体
            for entity in entities:
                if entity.id not in graph.entities:
                    graph.entities[entity.id] = entity
                else:
                    existing = graph.entities[entity.id]
                    if document_id not in existing.document_ids:
                        existing.document_ids.append(document_id)

            # 合并关系（去重）
            for relation in relations:
                if relation.id in existing_rel_ids:
                    continue
                existing_rel_ids.add(relation.id)
                graph.relations.append(relation)

        # 如果所有抽取任务都失败：不要“静默成功”为 0/0，直接抛错提示用户检查模型配置/API。
        if results and err_count == len(results):
            raise RuntimeError("知识图谱构建失败：LLM 调用全部失败。请检查“设置→模型选择”的厂商/模型，以及对应 API Key/网络。")

        # 如果部分失败且最终图谱仍为空，也不要“静默成功”
        if results and err_count > 0 and (len(graph.entities) == 0 and len(graph.relations) == 0):
            raise RuntimeError("知识图谱构建失败：LLM 调用出现错误且结果为空。建议检查模型超时/输出长度配置。")

        # 5) 写入缓存（内存 + SQLite）
        self._graph_cache[mem_key] = graph
        self._save_graph_to_persistent_cache(
            user_id=user_id,
            graph_key=graph_key,
            source_hash=source_hash,
            graph=graph,
            document_ids=[document_id],
            scope="document",
            meta={
                "cache_version": self.CACHE_VERSION,
                "parent_chunks": len(chunks),
                "merged_batches": len(merged_texts),
            },
        )
        return graph
    
    def build_from_documents(
        self,
        document_ids: List[str],
        user_id: str,
        align_entities: bool = True
    ) -> KnowledgeGraph:
        """同步构建多文档知识图谱（内部走异步并发）。"""
        return self._run_async(
            self.build_from_documents_async(
                document_ids=document_ids,
                user_id=user_id,
                align_entities=align_entities,
            )
        )

    async def build_from_documents_async(
        self,
        document_ids: List[str],
        user_id: str,
        align_entities: bool = True,
    ) -> KnowledgeGraph:
        """异步构建多文档知识图谱：并发构建子图 + 合并 +（可选）实体对齐 + 持久化缓存。"""
        doc_ids = sorted(set(document_ids or []))
        if not doc_ids:
            return KnowledgeGraph()

        llm_fp = self._llm_fingerprint(user_id)
        graph_key = self._graph_key_for_documents(doc_ids, align_entities=align_entities, llm_fp=llm_fp)
        mem_key = self._mem_cache_key(user_id, graph_key)

        # 1) 内存缓存
        if mem_key in self._graph_cache:
            cached_mem = self._graph_cache[mem_key]
            if cached_mem.entities or cached_mem.relations:
                return cached_mem

        # 2) 计算全量 source hash（用于缓存失效判断）
        h = hashlib.sha256()
        for did in doc_ids:
            chunks = self._get_parent_chunks_for_kg(did)
            h.update(did.encode("utf-8"))
            h.update(b"\0")
            h.update(self._hash_chunks(chunks).encode("utf-8"))
            h.update(b"\0")
        source_hash = h.hexdigest()

        # 3) 持久化缓存
        cached = self._load_graph_from_persistent_cache(user_id=user_id, graph_key=graph_key, source_hash=source_hash)
        if cached is not None:
            self._graph_cache[mem_key] = cached
            return cached

        # 单文档“全部文档”场景：直接复用单文档构建结果（避免不必要的对齐与多文档缓存被空结果污染）
        if len(doc_ids) == 1:
            ctx = self._new_async_context()
            g = await self.build_from_document_async(document_id=doc_ids[0], user_id=user_id, ctx=ctx)
            # 写入 multi-doc cache key（即便只有一个文档，也能让“全部文档”下次秒开）
            self._graph_cache[mem_key] = g
            self._save_graph_to_persistent_cache(
                user_id=user_id,
                graph_key=graph_key,
                source_hash=source_hash,
                graph=g,
                document_ids=doc_ids,
                scope="multi_document",
                meta={
                    "cache_version": self.CACHE_VERSION,
                    "documents": len(doc_ids),
                    "aligned": bool(align_entities),
                    "single_doc_shortcut": True,
                },
            )
            return g

        # 4) 并发构建每个文档子图（共享 ctx：统一限流/并发）
        ctx = self._new_async_context()
        tasks = [self.build_from_document_async(document_id=did, user_id=user_id, ctx=ctx) for did in doc_ids]
        doc_graphs = await asyncio.gather(*tasks, return_exceptions=True)

        merged_graph = KnowledgeGraph()
        existing_rel_ids: Set[str] = set()

        for g in doc_graphs:
            if isinstance(g, Exception):
                continue

            # 合并实体
            for entity_id, entity in g.entities.items():
                if entity_id not in merged_graph.entities:
                    merged_graph.entities[entity_id] = entity
                else:
                    existing = merged_graph.entities[entity_id]
                    for did in entity.document_ids:
                        if did not in existing.document_ids:
                            existing.document_ids.append(did)

            # 合并关系（去重）
            for relation in g.relations:
                if relation.id in existing_rel_ids:
                    continue
                existing_rel_ids.add(relation.id)
                merged_graph.relations.append(relation)

        # 5) 实体对齐（可选）
        if align_entities and len(merged_graph.entities) > 1:
            merged_graph = await self._align_entities_async(merged_graph, ctx=ctx, user_id=user_id)

        # 6) 写入缓存（内存 + SQLite）
        self._graph_cache[mem_key] = merged_graph
        self._save_graph_to_persistent_cache(
            user_id=user_id,
            graph_key=graph_key,
            source_hash=source_hash,
            graph=merged_graph,
            document_ids=doc_ids,
            scope="multi_document",
            meta={
                "cache_version": self.CACHE_VERSION,
                "documents": len(doc_ids),
                "aligned": bool(align_entities),
                "alignment_direct_threshold": self.alignment_direct_threshold,
            },
        )

        return merged_graph
    
    async def _llm_entity_match_async(self, e1: Entity, e2: Entity, ctx: _KGAsyncContext, user_id: str) -> bool:
        """使用 LLM 判断两个实体是否相同（带限流/重试）。"""
        prompt = self.ALIGNMENT_PROMPT.format(
            name_a=e1.name,
            type_a=e1.type.value,
            desc_a=e1.description,
            name_b=e2.name,
            type_b=e2.type.value,
            desc_b=e2.description,
        )

        messages = [HumanMessage(content=prompt)]
        response = await self._ainvoke_with_retry(messages, ctx, user_id=user_id)
        return "SAME" in (getattr(response, "content", "") or "").upper()

    async def _align_entities_async(self, graph: KnowledgeGraph, ctx: _KGAsyncContext, user_id: str) -> KnowledgeGraph:
        """
        实体对齐：合并指向同一概念的不同实体（异步：LLM 判断带限流）。

        策略：
        1) embedding 相似度 >= alignment_direct_threshold（默认 0.80）直接合并
        2) alignment_llm_threshold <= 相似度 < alignment_direct_threshold 用 LLM 判断（但总次数做上限）
        """
        entities = list(graph.entities.values())
        if len(entities) < 2:
            return graph

        # 获取所有实体的 embeddings（同步调用，带重试）
        texts = [f"{e.name}: {e.description}" for e in entities]
        embeddings = self._embed_documents_with_retry(texts)
        for i, entity in enumerate(entities):
            entity.embedding = embeddings[i]

        merge_pairs: List[Tuple[Entity, Entity]] = []
        merged_ids: Set[str] = set()
        llm_checks = 0

        for i, e1 in enumerate(entities):
            if e1.id in merged_ids:
                continue
            for e2 in entities[i + 1 :]:
                if e2.id in merged_ids:
                    continue

                similarity = self._cosine_similarity(e1.embedding or [], e2.embedding or [])

                if similarity >= self.alignment_direct_threshold:
                    merge_pairs.append((e1, e2))
                    merged_ids.add(e2.id)
                    continue

                if similarity >= self.alignment_llm_threshold and llm_checks < self.alignment_max_llm_checks:
                    llm_checks += 1
                    try:
                        if await self._llm_entity_match_async(e1, e2, ctx, user_id=user_id):
                            merge_pairs.append((e1, e2))
                            merged_ids.add(e2.id)
                    except Exception:
                        # LLM 判断失败就跳过，不影响主流程
                        continue

        # 执行合并
        for keep, remove in merge_pairs:
            # 合并 document_ids
            for did in remove.document_ids:
                if did not in keep.document_ids:
                    keep.document_ids.append(did)

            # 更新关系中的引用
            for relation in graph.relations:
                if relation.source_id == remove.id:
                    relation.source_id = keep.id
                if relation.target_id == remove.id:
                    relation.target_id = keep.id

            # 删除被合并的实体
            graph.entities.pop(remove.id, None)

        # 去重关系
        unique_relations: Dict[Tuple[str, str, RelationType], Relation] = {}
        for r in graph.relations:
            key = (r.source_id, r.target_id, r.type)
            if key not in unique_relations:
                unique_relations[key] = r
        graph.relations = list(unique_relations.values())

        return graph
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """计算余弦相似度"""
        import math
        if not a or not b:
            return 0.0
        # 避免不同 embedding 维度（例如缓存/配置变更）导致相似度“看似正常但不可靠”
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)
    
    # 保留旧名字，避免外部调用报错（同步包装）
    def _align_entities(self, graph: KnowledgeGraph) -> KnowledgeGraph:
        async def _runner():
            ctx = self._new_async_context()
            # 无 user_id 场景使用默认模型
            return await self._align_entities_async(graph, ctx=ctx, user_id="")

        return self._run_async(_runner())
    
    def get_entity_neighbors(
        self,
        entity_id: str,
        graph: KnowledgeGraph,
        depth: int = 1
    ) -> KnowledgeGraph:
        """
        获取实体的邻居子图
        
        Args:
            entity_id: 实体ID
            graph: 完整知识图谱
            depth: 搜索深度
            
        Returns:
            KnowledgeGraph: 子图
        """
        if entity_id not in graph.entities:
            return KnowledgeGraph()
        
        visited_entities = {entity_id}
        current_level = {entity_id}
        
        for _ in range(depth):
            next_level = set()
            for eid in current_level:
                # 找出所有相关关系
                for relation in graph.relations:
                    if relation.source_id == eid:
                        next_level.add(relation.target_id)
                    elif relation.target_id == eid:
                        next_level.add(relation.source_id)
            
            new_entities = next_level - visited_entities
            visited_entities.update(new_entities)
            current_level = new_entities
        
        # 构建子图
        subgraph = KnowledgeGraph()
        for eid in visited_entities:
            if eid in graph.entities:
                subgraph.entities[eid] = graph.entities[eid]
        
        for relation in graph.relations:
            if relation.source_id in visited_entities and relation.target_id in visited_entities:
                subgraph.relations.append(relation)
        
        return subgraph
    
    def search_entities(
        self,
        query: str,
        graph: KnowledgeGraph,
        top_k: int = 5
    ) -> List[Entity]:
        """
        搜索相关实体
        
        Args:
            query: 搜索查询
            graph: 知识图谱
            top_k: 返回数量
            
        Returns:
            List[Entity]: 匹配的实体列表
        """
        try:
            top_k = int(top_k)
        except Exception:
            top_k = 5
        if top_k <= 0:
            return []

        query = (query or "").strip()
        if not query or not graph.entities:
            return []
        
        entities = list(graph.entities.values())

        def _normalize(s: str) -> str:
            s = unicodedata.normalize("NFKC", str(s or "")).lower().strip()
            # 去掉常见噪声符号（保留中文、字母、数字、希腊字母等）
            s = re.sub(r"\s+", "", s)
            s = re.sub(r"[`~!@#$%^&*()\-_=+\[\]{}\\|;:'\",.<>/?，。、《》【】（）“”‘’！？·]+", "", s)
            # 处理常见 LaTeX 包裹
            s = s.replace("$", "")
            return s

        def _char_ngrams(s: str, n: int) -> Set[str]:
            s = s or ""
            if not s:
                return set()
            if len(s) <= n:
                return {s}
            return {s[i : i + n] for i in range(0, len(s) - n + 1)}

        def _lexical_score(q: str, t: str) -> float:
            if not q or not t:
                return 0.0
            if q == t:
                return 1.0
            if q in t:
                ratio = len(q) / max(1, len(t))
                return 0.85 + 0.15 * ratio
            if t in q:
                ratio = len(t) / max(1, len(q))
                return 0.80 + 0.20 * ratio

            # difflib 对中英文都能工作，但对“短 query”更稳定；n-gram Jaccard 对中文更鲁棒
            seq = difflib.SequenceMatcher(None, q, t).ratio()
            n2_q = _char_ngrams(q, 2)
            n2_t = _char_ngrams(t, 2)
            if n2_q and n2_t:
                jac = len(n2_q & n2_t) / len(n2_q | n2_t)
            else:
                jac = 0.0
            return max(seq, jac)

        q_norm = _normalize(query)
        if not q_norm:
            return []

        # 获取查询的 embedding（失败则退化为纯文本匹配）
        query_embedding: Optional[List[float]] = None
        try:
            query_embedding = self._embed_query_with_retry(query)
        except Exception:
            query_embedding = None

        # 确保所有实体都有 embedding（只有在 query_embedding 可用时才做；否则不触发网络调用）
        if query_embedding is not None:
            expected_dim = len(query_embedding)
            entities_without_embedding = [
                e
                for e in entities
                if (not e.embedding) or (expected_dim > 0 and len(e.embedding) != expected_dim)
            ]
            if entities_without_embedding:
                texts = [f"{e.name}: {e.description}" for e in entities_without_embedding]
                try:
                    embeddings = self._embed_documents_with_retry(texts)
                    for e, emb in zip(entities_without_embedding, embeddings):
                        e.embedding = emb
                except Exception:
                    query_embedding = None

        scored: List[Tuple[Entity, float]] = []
        for entity in entities:
            name_norm = _normalize(entity.name)
            desc_norm = _normalize(entity.description)

            lex_name = _lexical_score(q_norm, name_norm)
            lex_desc = _lexical_score(q_norm, desc_norm) * 0.85
            lex = max(lex_name, lex_desc)

            emb_sim = 0.0
            if query_embedding is not None and entity.embedding is not None:
                try:
                    emb_sim = self._cosine_similarity(query_embedding, entity.embedding or [])
                except Exception:
                    emb_sim = 0.0
                if emb_sim < 0:
                    emb_sim = 0.0

            # Hybrid: embedding 为主，文本匹配做强约束补偿（确保精确/包含匹配排在前面）
            score = 0.65 * emb_sim + 0.35 * lex

            # 轻量类型偏好：当 query 自带“定理/公式/方法”等线索时，优先同类实体
            try:
                q_raw = query
                if ("定理" in q_raw) or ("定律" in q_raw):
                    if entity.type == EntityType.THEOREM:
                        score += 0.10
                    elif entity.type == EntityType.CONCEPT:
                        score += 0.04
                if ("公式" in q_raw) or ("方程" in q_raw) or ("=" in q_raw):
                    if entity.type == EntityType.FORMULA:
                        score += 0.10
                if ("方法" in q_raw) or ("算法" in q_raw) or ("步骤" in q_raw):
                    if entity.type == EntityType.METHOD:
                        score += 0.08
                if ("例" in q_raw) or ("示例" in q_raw) or ("案例" in q_raw):
                    if entity.type == EntityType.EXAMPLE:
                        score += 0.06
            except Exception:
                pass

            if q_norm == name_norm:
                score += 1.0
            elif q_norm and name_norm.startswith(q_norm):
                score += 0.35
            elif q_norm and q_norm in name_norm:
                score += 0.25
            elif q_norm and q_norm in desc_norm:
                score += 0.10

            scored.append((entity, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]
    
    def clear_cache(
        self,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        persistent: bool = False,
    ):
        """清除缓存（内存缓存 + 可选持久化缓存）。"""
        if user_id and document_id:
            # 单文档缓存 key 会包含模型指纹，清理时按前缀清空该文档的所有模型版本
            prefix = f"{user_id}|kg:doc:v{self.CACHE_VERSION}:{document_id}:"
            keys_to_remove = [k for k in self._graph_cache if k.startswith(prefix)]
            for k in keys_to_remove:
                self._graph_cache.pop(k, None)
            if persistent:
                self.db.delete_kg_cache_by_document(user_id=user_id, document_id=document_id)
            return

        if user_id:
            prefix = f"{user_id}|"
            keys_to_remove = [k for k in self._graph_cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._graph_cache[k]
            # 持久化按用户清理暂不做（避免误删）；需要的话可在 DB 层补充 delete_kg_cache_by_user
            return

        # 全量清理（仅清内存缓存；避免误删所有用户的持久化缓存）
        self._graph_cache.clear()
        return
    
    def _build_graph_from_extraction_result(
        self,
        result: Dict,
        document_id: str
    ) -> KnowledgeGraph:
        """从抽取结果构建知识图谱"""
        graph = KnowledgeGraph()
        entity_name_to_id: Dict[str, str] = {}
        
        # 处理实体
        for e in result.get("entities", []) or []:
            if not isinstance(e, dict):
                continue
            name = (e.get("name") or "").strip()
            if not name:
                continue
            
            raw_type = (e.get("type") or EntityType.CONCEPT.value).strip()
            try:
                entity_type = EntityType(raw_type)
            except Exception:
                entity_type = EntityType.CONCEPT
                raw_type = entity_type.value
            
            entity_id = self._generate_entity_id(name, raw_type)
            entity = Entity(
                id=entity_id,
                name=name,
                type=entity_type,
                description=(e.get("description") or "").strip(),
                document_ids=[document_id],
            )
            graph.entities[entity_id] = entity
            entity_name_to_id[name] = entity_id
        
        # 处理关系
        existing_rel_ids: Set[str] = set()
        for r in result.get("relations", []) or []:
            if not isinstance(r, dict):
                continue
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            if not src or not tgt:
                continue
            
            source_id = entity_name_to_id.get(src)
            target_id = entity_name_to_id.get(tgt)
            if not source_id or not target_id:
                continue
            
            raw_type = (r.get("type") or RelationType.LEADS_TO.value).strip()
            try:
                rel_type = RelationType(raw_type)
            except Exception:
                rel_type = RelationType.LEADS_TO
                raw_type = rel_type.value
            
            rel_id = self._generate_relation_id(source_id, target_id, raw_type)
            if rel_id in existing_rel_ids:
                continue
            existing_rel_ids.add(rel_id)
            
            relation = Relation(
                id=rel_id,
                source_id=source_id,
                target_id=target_id,
                type=rel_type,
                description=(r.get("description") or None),
                document_ids=[document_id],
            )
            graph.relations.append(relation)
        
        return graph
    
    def build_global_graph_fast(
        self,
        user_id: str,
        save_snapshot: bool = True,
        max_chars_per_doc: int = 8000
    ) -> Tuple[KnowledgeGraph, Optional[str]]:
        """
        快速构建用户所有文档的全局知识图谱
        
        Args:
            user_id: 用户ID
            save_snapshot: 是否自动保存快照
            max_chars_per_doc: 每个文档的最大字符数
            
        Returns:
            Tuple[KnowledgeGraph, Optional[str]]: (图谱对象, 快照文件路径或None)
        """
        # 获取用户所有已完成处理的文档
        docs = self.db.get_user_documents(user_id, status="completed")
        if not docs:
            return KnowledgeGraph(), None
        
        # 收集所有文档的文本
        all_texts = []
        doc_ids = []
        
        for doc in docs:
            chunks = self._get_parent_chunks_for_kg(doc.id)
            if chunks:
                doc_text = "\n".join([getattr(c, "content", "") for c in chunks])
                if len(doc_text) > max_chars_per_doc:
                    doc_text = doc_text[:max_chars_per_doc]
                all_texts.append(f"## 文档: {doc.filename}\n{doc_text}")
                doc_ids.append(doc.id)
        
        if not all_texts:
            return KnowledgeGraph(), None
        
        # 合并所有文本
        combined_text = "\n\n===\n\n".join(all_texts)
        
        # 限制总长度
        max_total_chars = 20000
        if len(combined_text) > max_total_chars:
            combined_text = combined_text[:max_total_chars] + "\n\n[内容已截断...]"
        
        # 一次 LLM 调用
        prompt = self.BATCH_EXTRACTION_PROMPT.format(text=combined_text)
        messages = [HumanMessage(content=prompt)]
        
        try:
            llm = self._get_llm(user_id=user_id)
            response = llm.invoke(messages)
            result = self._parse_extraction_response(getattr(response, "content", "") or "")
        except Exception as e:
            print(f"LLM 调用失败: {e}")
            return KnowledgeGraph(), None
        
        # 构建图谱（使用第一个文档ID作为来源标记）
        primary_doc_id = doc_ids[0] if doc_ids else "unknown"
        graph = self._build_graph_from_extraction_result(result, primary_doc_id)
        
        # 更新所有实体的 document_ids
        for entity in graph.entities.values():
            entity.document_ids = doc_ids
        for relation in graph.relations:
            relation.document_ids = doc_ids
        
        # 保存快照
        snapshot_path = None
        if save_snapshot and len(graph.entities) > 0:
            snapshot_path = self.save_graph_snapshot(graph)
        
        return graph, snapshot_path

    def build_global_graph(
        self,
        user_id: str,
        align_entities: bool = True,
        save_snapshot: bool = True
    ) -> Tuple[KnowledgeGraph, Optional[str]]:
        """
        构建用户所有文档的全局知识图谱
        
        Args:
            user_id: 用户ID
            align_entities: 是否进行实体对齐
            save_snapshot: 是否自动保存快照
            
        Returns:
            Tuple[KnowledgeGraph, Optional[str]]: (图谱对象, 快照文件路径或None)
        """
        # 获取用户所有已完成处理的文档
        docs = self.db.get_user_documents(user_id, status="completed")
        if not docs:
            return KnowledgeGraph(), None
        
        doc_ids = [doc.id for doc in docs]
        
        # 构建多文档图谱
        graph = self.build_from_documents(
            document_ids=doc_ids,
            user_id=user_id,
            align_entities=align_entities
        )
        
        # 保存快照
        snapshot_path = None
        if save_snapshot and len(graph.entities) > 0:
            snapshot_path = self.save_graph_snapshot(graph)
        
        return graph, snapshot_path
