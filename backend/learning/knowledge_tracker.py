"""
知识追踪模块
实现用户知识状态管理、薄弱点识别和个性化出题
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from ..database.crud import DatabaseManager
from ..database.models import KnowledgeState


@dataclass
class KnowledgePoint:
    """知识点数据结构"""
    name: str
    attempts: int
    correct_count: int
    mastery_rate: float
    is_weak_point: bool
    last_attempt: Optional[datetime]
    weight: float  # 出题权重


class KnowledgeTracker:
    """知识追踪器"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        weak_threshold: float = 0.6,
        min_attempts_for_weak: int = 3,
        forgetting_factor: float = 0.1
    ):
        """
        初始化知识追踪器
        
        Args:
            db_manager: 数据库管理器
            weak_threshold: 薄弱点阈值（掌握率低于此值视为薄弱）
            min_attempts_for_weak: 判定薄弱点的最小尝试次数
            forgetting_factor: 遗忘因子（用于长时间未复习的知识点）
        """
        self.db = db_manager
        self.weak_threshold = weak_threshold
        self.min_attempts_for_weak = min_attempts_for_weak
        self.forgetting_factor = forgetting_factor

    @staticmethod
    def _proficiency_snapshot(state: KnowledgeState) -> Tuple[float, str, str]:
        """
        将掌握情况映射为“熟练度”与状态标签
        
        - 熟练度：以正确次数 * 10 作为经验值上限 100，不扣分
        - 状态：red < 40, yellow 40-69, green >= 70
        - 标签：待复习 / 需巩固 / 已点亮
        """
        xp = min(100.0, state.correct_count * 10.0)
        if xp >= 70:
            status, label = "green", "已点亮"
        elif xp >= 40:
            status, label = "yellow", "需巩固"
        else:
            status, label = "red", "待复习"
        return xp, status, label

    @staticmethod
    def _categorize_point(name: str) -> str:
        """根据知识点名称粗略归类到代数/几何/微积分/其他。"""
        lower = name.lower()
        algebra_keys = ["方程", "函数", "代数", "多项式", "数", "实数", "有理", "无理"]
        geometry_keys = ["几何", "三角", "角", "圆", "面积", "体积", "直线", "平面", "图形"]
        calculus_keys = ["微分", "积分", "导数", "极限", "微积分"]
        if any(k in lower or k in name for k in calculus_keys):
            return "微积分"
        if any(k in lower or k in name for k in geometry_keys):
            return "几何"
        if any(k in lower or k in name for k in algebra_keys):
            return "代数"
        return "其他"
    
    def update_knowledge_state(
        self,
        user_id: str,
        knowledge_point: str,
        is_correct: bool
    ) -> KnowledgePoint:
        """
        更新用户对某知识点的掌握状态
        
        Args:
            user_id: 用户ID
            knowledge_point: 知识点名称
            is_correct: 本次答题是否正确
            
        Returns:
            KnowledgePoint: 更新后的知识点状态
        """
        state = self.db.update_knowledge_state(user_id, knowledge_point, is_correct)
        
        return KnowledgePoint(
            name=state.knowledge_point,
            attempts=state.attempts,
            correct_count=state.correct_count,
            mastery_rate=state.mastery_rate,
            is_weak_point=state.is_weak_point,
            last_attempt=state.last_attempt,
            weight=self._calculate_weight(state)
        )
    
    def _calculate_weight(self, state: KnowledgeState) -> float:
        """
        计算知识点的出题权重
        
        权重公式：weight = base_weight * time_decay
        - base_weight = 1 / (mastery_rate + epsilon)
        - time_decay = 根据上次练习时间的衰减因子
        
        Args:
            state: 知识状态对象
            
        Returns:
            float: 出题权重
        """
        epsilon = 0.1  # 防止除零
        
        # 基础权重：掌握率越低，权重越高
        base_weight = 1.0 / (state.mastery_rate + epsilon)
        
        # 时间衰减：长时间未复习的知识点权重增加
        time_decay = 1.0
        if state.last_attempt:
            days_since_last = (datetime.utcnow() - state.last_attempt).days
            # 超过7天未复习，权重开始增加
            if days_since_last > 7:
                time_decay = 1.0 + self.forgetting_factor * (days_since_last - 7) / 7
        
        return base_weight * time_decay
    
    def get_user_knowledge_map(self, user_id: str) -> Dict[str, KnowledgePoint]:
        """
        获取用户的完整知识图谱
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, KnowledgePoint]: 知识点名称到状态的映射
        """
        states = self.db.get_user_knowledge_states(user_id)
        
        knowledge_map = {}
        for state in states:
            knowledge_map[state.knowledge_point] = KnowledgePoint(
                name=state.knowledge_point,
                attempts=state.attempts,
                correct_count=state.correct_count,
                mastery_rate=state.mastery_rate,
                is_weak_point=state.is_weak_point,
                last_attempt=state.last_attempt,
                weight=self._calculate_weight(state)
            )
        
        return knowledge_map
    
    def get_weak_points(self, user_id: str) -> List[KnowledgePoint]:
        """
        获取用户的薄弱知识点
        
        Args:
            user_id: 用户ID
            
        Returns:
            List[KnowledgePoint]: 薄弱知识点列表（按掌握率升序排列）
        """
        states = self.db.get_user_weak_points(user_id)
        
        weak_points = []
        for state in states:
            weak_points.append(KnowledgePoint(
                name=state.knowledge_point,
                attempts=state.attempts,
                correct_count=state.correct_count,
                mastery_rate=state.mastery_rate,
                is_weak_point=True,
                last_attempt=state.last_attempt,
                weight=self._calculate_weight(state)
            ))
        
        return weak_points
    
    def get_recommended_knowledge_points(
        self,
        user_id: str,
        num_points: int = 5,
        include_weak: bool = True,
        include_forgotten: bool = True
    ) -> List[str]:
        """
        获取推荐复习的知识点
        
        Args:
            user_id: 用户ID
            num_points: 推荐数量
            include_weak: 是否包含薄弱点
            include_forgotten: 是否包含长时间未复习的
            
        Returns:
            List[str]: 推荐的知识点名称列表
        """
        knowledge_map = self.get_user_knowledge_map(user_id)
        
        if not knowledge_map:
            return []
        
        # 计算综合推荐分数
        scored_points = []
        for name, point in knowledge_map.items():
            score = 0.0
            
            # 薄弱点加分
            if include_weak and point.is_weak_point:
                score += 5.0
            
            # 低掌握率加分
            score += (1 - point.mastery_rate) * 3.0
            
            # 长时间未复习加分
            if include_forgotten and point.last_attempt:
                days_since = (datetime.utcnow() - point.last_attempt).days
                if days_since > 7:
                    score += min(days_since / 7, 3.0)
            
            scored_points.append((name, score))
        
        # 按分数排序
        scored_points.sort(key=lambda x: -x[1])
        
        return [name for name, _ in scored_points[:num_points]]
    
    def weighted_sample_knowledge_points(
        self,
        user_id: str,
        available_points: List[str],
        num_samples: int = 3
    ) -> List[str]:
        """
        按权重采样知识点（用于个性化出题）
        
        Args:
            user_id: 用户ID
            available_points: 可用的知识点列表
            num_samples: 采样数量
            
        Returns:
            List[str]: 采样的知识点列表
        """
        if not available_points:
            return []
        
        if len(available_points) <= num_samples:
            return available_points
        
        knowledge_map = self.get_user_knowledge_map(user_id)
        
        # 计算权重
        weights = []
        for point in available_points:
            if point in knowledge_map:
                weights.append(knowledge_map[point].weight)
            else:
                # 新知识点给予中等权重
                weights.append(1.0)
        
        # 归一化权重
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 加权随机采样
        sampled = []
        remaining_points = list(available_points)
        remaining_weights = list(normalized_weights)
        
        for _ in range(min(num_samples, len(remaining_points))):
            # 按权重选择
            r = random.random()
            cumsum = 0
            for i, w in enumerate(remaining_weights):
                cumsum += w
                if r <= cumsum:
                    sampled.append(remaining_points[i])
                    remaining_points.pop(i)
                    remaining_weights.pop(i)
                    # 重新归一化
                    if remaining_weights:
                        total = sum(remaining_weights)
                        remaining_weights = [w / total for w in remaining_weights]
                    break
        
        return sampled
    
    def get_learning_statistics(self, user_id: str) -> Dict[str, Any]:
        """
        获取用户学习统计数据
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict: 学习统计数据
        """
        knowledge_map = self.get_user_knowledge_map(user_id)
        weak_points = self.get_weak_points(user_id)
        
        if not knowledge_map:
            return {
                "total_knowledge_points": 0,
                "total_attempts": 0,
                "overall_accuracy": 0.0,
                "weak_points_count": 0,
                "mastered_count": 0,
                "learning_count": 0,
                "average_mastery": 0.0,
                "proficiency_points": [],
                "top_weak_points": [],
                "tag_wall": {"green": [], "yellow": [], "red": []},
                "lit_count": 0,
                "recently_mastered": [],
                "radar": [],
            }
        
        total_attempts = sum(p.attempts for p in knowledge_map.values())
        total_correct = sum(p.correct_count for p in knowledge_map.values())
        
        mastered = [p for p in knowledge_map.values() if p.mastery_rate >= 0.8]
        learning = [p for p in knowledge_map.values() if 0.4 <= p.mastery_rate < 0.8]

        proficiency_points = []
        tag_wall = {"green": [], "yellow": [], "red": []}
        radar_bucket: Dict[str, List[float]] = {"代数": [], "几何": [], "微积分": [], "其他": []}
        for state in knowledge_map.values():
            xp, status, label = self._proficiency_snapshot(
                KnowledgeState(
                    user_id=user_id,
                    knowledge_point=state.name,
                    attempts=state.attempts,
                    correct_count=state.correct_count,
                    mastery_rate=state.mastery_rate,
                    last_attempt=state.last_attempt,
                    is_weak_point=state.is_weak_point,
                )
            )
            item = {
                "name": state.name,
                "xp": xp,
                "status": status,
                "label": label,
                "mastery_rate": state.mastery_rate,
                "attempts": state.attempts,
                "last_attempt": state.last_attempt,
            }
            proficiency_points.append(item)
            tag_wall[status].append(state.name)
            radar_bucket[self._categorize_point(state.name)].append(xp)

        top_weak_points = sorted(proficiency_points, key=lambda x: x["xp"])[:5]
        radar = []
        for cat, vals in radar_bucket.items():
            radar.append({
                "category": cat,
                "xp": sum(vals) / len(vals) if vals else 0.0
            })
        
        return {
            "total_knowledge_points": len(knowledge_map),
            "total_attempts": total_attempts,
            "overall_accuracy": total_correct / total_attempts if total_attempts > 0 else 0.0,
            "weak_points_count": len(weak_points),
            "mastered_count": len(mastered),
            "learning_count": len(learning),
            "average_mastery": sum(p.mastery_rate for p in knowledge_map.values()) / len(knowledge_map),
            "weak_points": [p.name for p in weak_points[:5]],
            "recently_mastered": [p.name for p in mastered if p.last_attempt and 
                                  (datetime.utcnow() - p.last_attempt).days < 7][:5],
            "proficiency_points": sorted(proficiency_points, key=lambda x: x["xp"], reverse=True),
            "top_weak_points": top_weak_points,
            "tag_wall": tag_wall,
            "lit_count": len(tag_wall["green"]),
            "radar": radar,
        }
    
    def generate_study_plan(
        self,
        user_id: str,
        target_days: int = 7
    ) -> Dict[str, Any]:
        """
        生成学习计划建议
        
        Args:
            user_id: 用户ID
            target_days: 计划天数
            
        Returns:
            Dict: 学习计划
        """
        weak_points = self.get_weak_points(user_id)
        recommended = self.get_recommended_knowledge_points(user_id, num_points=10)
        stats = self.get_learning_statistics(user_id)
        
        # 分配知识点到各天
        daily_plan = {}
        points_per_day = max(1, len(recommended) // target_days)
        
        for day in range(target_days):
            start_idx = day * points_per_day
            end_idx = start_idx + points_per_day
            daily_points = recommended[start_idx:end_idx]
            
            if daily_points:
                daily_plan[f"第{day + 1}天"] = {
                    "knowledge_points": daily_points,
                    "focus": "复习薄弱点" if any(p in [w.name for w in weak_points] for p in daily_points) else "巩固练习"
                }
        
        return {
            "summary": f"您有 {len(weak_points)} 个薄弱知识点需要加强",
            "overall_mastery": f"{stats['average_mastery'] * 100:.1f}%",
            "daily_plan": daily_plan,
            "priority_points": [p.name for p in weak_points[:3]] if weak_points else [],
            "estimated_improvement": "预计完成计划后掌握率可提升10-15%"
        }
