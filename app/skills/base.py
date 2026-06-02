from abc import ABC, abstractmethod
from langchain.tools import Tool


class BaseSkill(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, query: str) -> str: ...

    def as_tool(self) -> Tool:
        return Tool(name=self.name, func=self.run, description=self.description)
