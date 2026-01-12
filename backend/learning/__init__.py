"""学习功能模块

注意：这里不要做“全量 eager import”。Streamlit 多页面可能在不同线程中并发加载模块，
eager import 容易触发 Python import lock 的死锁检测（_DeadlockError）。

如需从包级别导入（例如 `from backend.learning import KnowledgeGraphBuilder`），
将通过 `__getattr__` 延迟加载对应子模块。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "QuizGenerator",
    "KnowledgeTracker",
    "SocraticEngine",
    "KnowledgeGraphBuilder",
    "KnowledgeGraph",
    "Entity",
    "Relation",
    "EntityType",
    "RelationType",
    "QuizQuestion",
]


def __getattr__(name: str) -> Any:
    if name in {"QuizGenerator", "QuizQuestion"}:
        from .quiz_generator import QuizGenerator, QuizQuestion

        return QuizGenerator if name == "QuizGenerator" else QuizQuestion

    if name == "KnowledgeTracker":
        from .knowledge_tracker import KnowledgeTracker

        return KnowledgeTracker

    if name == "SocraticEngine":
        from .socratic_engine import SocraticEngine

        return SocraticEngine

    if name in {
        "KnowledgeGraphBuilder",
        "KnowledgeGraph",
        "Entity",
        "Relation",
        "EntityType",
        "RelationType",
    }:
        from .kg_builder import (
            Entity,
            EntityType,
            KnowledgeGraph,
            KnowledgeGraphBuilder,
            Relation,
            RelationType,
        )

        return {
            "KnowledgeGraphBuilder": KnowledgeGraphBuilder,
            "KnowledgeGraph": KnowledgeGraph,
            "Entity": Entity,
            "Relation": Relation,
            "EntityType": EntityType,
            "RelationType": RelationType,
        }[name]

    raise AttributeError(name)
