"""Harness 调度层：包裹 AgentExecutor，统一返回格式、错误降级、耗时日志、评测钩子。

web_agent.py 只与本模块交互，不直接调用 AgentExecutor。
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 评测日志路径，可通过环境变量关闭/改路径
_EVAL_LOG = os.getenv("HARNESS_EVAL_LOG", os.path.join("data", "harness_eval.jsonl"))


@dataclass
class HarnessResult:
    output: str
    tool_used: str | None = None
    latency_ms: int = 0
    error: str | None = None


def _extract_tool(response: dict) -> str | None:
    """从 intermediate_steps 取最后一次命中的工具名。"""
    steps = response.get("intermediate_steps") or []
    if not steps:
        return None
    action = steps[-1][0]
    return getattr(action, "tool", None)


def _write_eval(query: str, result: HarnessResult) -> None:
    """把 (input, tool_used, output) 落盘成 jsonl，供 promptfoo 等离线评测使用。"""
    try:
        os.makedirs(os.path.dirname(_EVAL_LOG), exist_ok=True)
        record = {
            "input": query,
            "tool_used": result.tool_used,
            "output": result.output,
            "latency_ms": result.latency_ms,
            "error": result.error,
        }
        with open(_EVAL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:  # 评测日志失败不能影响主流程
        logger.warning(f"写评测日志失败: {e}")


def handle(query: str, agent_executor) -> HarnessResult:
    """统一入口：执行查询并返回结构化结果。任何异常都降级为友好提示，不向上抛。"""
    start = time.perf_counter()
    try:
        response = agent_executor.invoke({"input": query})
        result = HarnessResult(
            output=response.get("output", ""),
            tool_used=_extract_tool(response),
            latency_ms=int((time.perf_counter() - start) * 1000),
        )
    except Exception as e:
        logger.error(f"Harness 处理失败: {e}")
        result = HarnessResult(
            output="抱歉，处理你的请求时出了点问题，请稍后再试或换个说法～",
            latency_ms=int((time.perf_counter() - start) * 1000),
            error=str(e),
        )

    logger.info(f"[harness] tool={result.tool_used} latency={result.latency_ms}ms error={result.error}")
    _write_eval(query, result)
    return result
