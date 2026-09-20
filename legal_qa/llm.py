"""OpenAI 兼容的 LLM / Embedding 客户端。

设计目标：与部署形态解耦——同一份代码可用于
  1. AutoDL 上 vLLM 的 OpenAI 兼容服务（Qwen2.5-7B-Instruct）
  2. vLLM/TEI 的 embedding 服务（BGE-M3）
  3. 任何 OpenAI 兼容云端 API（调试用）

Skill 模块只依赖本文件的轻量接口，不直接依赖 agentscope，
便于独立单元测试与消融实验脚本复用。
"""
from __future__ import annotations

import json
import re
import time
from typing import List, Optional, Sequence

import numpy as np
import requests

from .config import EmbeddingConfig, LLMConfig


class LLMClient:
    """极简 OpenAI 兼容 Chat 客户端（带重试与 JSON 解析容错）。"""

    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()

    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: Sequence[dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retries: int = 2,
    ) -> str:
        """标准对话补全，返回助手文本。"""
        payload = {
            "model": self.cfg.model,
            "messages": list(messages),
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_tokens if max_tokens is None else max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        url = self.cfg.api_base_url.rstrip("/") + "/chat/completions"

        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = self._session.post(
                    url, json=payload, headers=headers, timeout=self.cfg.timeout
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001 - 网络层统一重试
                last_err = e
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"LLM 请求失败（{url}）: {last_err}") from last_err

    # ------------------------------------------------------------------ #
    def chat_json(
        self,
        messages: Sequence[dict],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
    ) -> dict:
        """要求模型输出 JSON 并解析。

        依次尝试：1) 直接 json.loads；2) 提取 ```json 代码块；
        3) 提取首个大括号平衡片段。均失败时抛 ValueError。
        """
        content = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        return parse_json_loose(content)


def parse_json_loose(content: str) -> dict:
    """从 LLM 输出中稳健地解析 JSON 对象。"""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    start = content.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(content)):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(content[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"无法从模型输出解析 JSON: {content[:200]}")


class EmbeddingClient:
    """OpenAI 兼容 embeddings 客户端。

    只需实现 `embed(texts) -> np.ndarray`，
    任何满足该签名的对象（含测试用桩）均可替换注入下游模块。
    """

    def __init__(self, cfg: EmbeddingConfig) -> None:
        self.cfg = cfg
        self._session = requests.Session()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.cfg.dim), dtype=np.float32)
        url = self.cfg.api_base_url.rstrip("/") + "/embeddings"
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"}
        resp = self._session.post(
            url,
            json={
                "model": self.cfg.model,
                "input": [str(t) for t in texts],
                "dimensions": self.cfg.dim,
            },
            headers=headers,
            timeout=self.cfg.timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        vectors = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        arr = np.asarray(vectors, dtype=np.float32)
        # L2 归一化，下游一律用内积做余弦相似度
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return arr / norm

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
