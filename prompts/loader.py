"""prompt 统一加载器。

所有 prompt 模板存放在 prompts/ 目录下的 .txt 文件中，
通过此模块加载，实现 prompt 与业务逻辑分离，方便独立迭代和 promptfoo 评测。
"""
import os

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__))

_cache: dict = {}


def load_prompt(name: str) -> str:
    """按名称加载 prompts/ 下对应的 .txt 文件，带内存缓存。

    Args:
        name: 文件名（不含扩展名），例如 'agent_system'、'title_optimize'

    Returns:
        prompt 字符串内容
    """
    if name not in _cache:
        path = os.path.join(_PROMPTS_DIR, f"{name}.txt")
        with open(path, encoding="utf-8") as f:
            _cache[name] = f.read()
    return _cache[name]
