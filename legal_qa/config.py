"""legal_qa 全局配置。

从 .env / 环境变量加载，对齐 AutoDL 部署：
  LLM       -> vLLM 的 OpenAI 兼容服务 (Qwen2.5-7B-Instruct)
  Embedding -> vLLM/TEI 的 OpenAI 兼容服务 (BGE-M3, 1024 维)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# legal_qa 包根目录（legal_qa/config.py 的上一级）
PACKAGE_ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Optional[str] = None) -> None:
    """极简 .env 加载器（不引入 python-dotenv 依赖）。"""
    if path is None:
        path = os.environ.get("LEGAL_QA_ENV", str(PACKAGE_ROOT / ".env"))
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class LLMConfig:
    """主 LLM（OpenAI 兼容，指向 vLLM）。"""
    api_base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen2.5-7B-Instruct"
    api_key: str = "EMPTY"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: float = 120.0


@dataclass
class EmbeddingConfig:
    """Embedding 服务（OpenAI 兼容，指向 vLLM serve BGE-M3 或 TEI）。"""
    api_base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "BGE-M3"
    api_key: str = "EMPTY"
    dim: int = 1024
    timeout: float = 60.0


@dataclass
class RetrievalConfig:
    """法条检索配置。"""
    index_dir: str = "data/law_index"   # 文件式向量索引（兼容 Legal-world 命名）
    top_k: int = 10
    rerank: bool = False               # 是否启用重排（bge-reranker 服务预留）


@dataclass
class DispatchConfig:
    """Skill 动态调度配置（论文创新点二）。

    policy:
      dynamic -> 基于记忆状态的动态调度（本文方法）
      fixed   -> 固定流程全量执行（消融实验对照组）
    """
    policy: str = "dynamic"
    fact_completeness_threshold: float = 0.6  # 事实完整度阈值，低于则触发追问
    memory_recall_top_k: int = 5               # 中期记忆召回条数
    max_dialogue_turns: int = 30               # 多轮对话安全上限


@dataclass
class AppConfig:
    """总配置。"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    dispatch: DispatchConfig = field(default_factory=DispatchConfig)
    data_dir: str = "data"
    output_dir: str = "outputs"
    seed: int = 42

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            llm=LLMConfig(
                api_base_url=_env("LLM_API_BASE_URL", "http://127.0.0.1:8000/v1"),
                model=_env("LLM_MODEL", "Qwen2.5-7B-Instruct"),
                api_key=_env("LLM_API_KEY", "EMPTY"),
                temperature=float(_env("LLM_TEMPERATURE", "0.1")),
                max_tokens=int(_env("LLM_MAX_TOKENS", "2048")),
                timeout=float(_env("LLM_TIMEOUT", "120")),
            ),
            embedding=EmbeddingConfig(
                api_base_url=_env("EMBED_API_BASE_URL", "http://127.0.0.1:8001/v1"),
                model=_env("EMBED_MODEL", "BGE-M3"),
                api_key=_env("EMBED_API_KEY", "EMPTY"),
                dim=int(_env("EMBED_DIM", "1024")),
                timeout=float(_env("EMBED_TIMEOUT", "60")),
            ),
            retrieval=RetrievalConfig(
                index_dir=_env("LAW_INDEX_DIR", "data/law_index"),
                top_k=int(_env("RETRIEVAL_TOP_K", "10")),
                rerank=_env("RETRIEVAL_RERANK", "false").lower() == "true",
            ),
            dispatch=DispatchConfig(
                policy=_env("DISPATCH_POLICY", "dynamic"),
                fact_completeness_threshold=float(_env("FACT_COMPLETENESS_THRESHOLD", "0.6")),
                memory_recall_top_k=int(_env("MEMORY_RECALL_TOP_K", "5")),
                max_dialogue_turns=int(_env("MAX_DIALOGUE_TURNS", "30")),
            ),
            data_dir=_env("DATA_DIR", "data"),
            output_dir=_env("OUTPUT_DIR", "outputs"),
            seed=int(_env("SEED", "42")),
        )

    def resolve_path(self, p: str) -> Path:
        """相对路径基于包根目录解析，便于在任意工作目录运行。"""
        path = Path(p)
        return path if path.is_absolute() else (PACKAGE_ROOT / path)
