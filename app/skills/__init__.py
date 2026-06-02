"""技能层：每个能力独立成类，统一 BaseSkill 接口，可独立实例化与单测。"""
from app.skills.base import BaseSkill
from app.skills.chat import ChatSkill
from app.skills.title_optimize import TitleOptimizeSkill
from app.skills.play_predict import PlayPredictSkill


def build_skills(llm) -> "list[BaseSkill]":
    """构造全部技能实例，依赖注入 llm。新增技能只需在此追加一行。"""
    return [
        ChatSkill(),
        TitleOptimizeSkill(llm),
        PlayPredictSkill(),
    ]


__all__ = [
    "BaseSkill",
    "ChatSkill",
    "TitleOptimizeSkill",
    "PlayPredictSkill",
    "build_skills",
]
