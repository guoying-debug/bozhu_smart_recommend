from langchain.prompts import PromptTemplate
from app.skills.base import BaseSkill
from app.core.recommender import get_similar_titles
from prompts.loader import load_prompt


class TitleOptimizeSkill(BaseSkill):
    name = "标题优化工具"
    description = "当用户需要优化B站视频标题时调用。输入用户提供的原始标题（可包含特殊要求）"

    def __init__(self, llm):
        self._llm = llm

    def run(self, query: str) -> str:
        similar = get_similar_titles(query, top_k=3)
        docs = "\n".join(f"- {t['title']} ({int(t['view_count']/10000)}万播放)" for t in similar)
        prompt = PromptTemplate(
            template=load_prompt("title_optimize"),
            input_variables=["docs", "query"]
        )
        return (prompt | self._llm).invoke({"docs": docs, "query": query}).content
