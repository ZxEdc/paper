"""法条向量检索（RAG 底座）。

沿用 Legal-world 的文件式向量索引设计（float16 npy + jsonl 元数据 + manifest），
使其发布的法条索引可直接放入本目录使用；同时提供从 STARD 语料自建索引的能力。

索引结构（LAW_INDEX_DIR）:
  law_vector_index_manifest.json  # {count, dim, model, created_at}
  law_embeddings.float16.npy       # [N, dim]，L2 归一化
  law_metadata.jsonl               # 每行 {doc_id, title, content, source}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .config import AppConfig

MANIFEST_NAME = "law_vector_index_manifest.json"
VECTOR_NAME = "law_embeddings.float16.npy"
METADATA_NAME = "law_metadata.jsonl"

EmbedFn = Callable[[List[str]], np.ndarray]  # texts -> [N, dim] L2 归一化


@dataclass
class LawArticle:
    doc_id: str
    title: str
    content: str
    source: str = ""

    def to_prompt(self) -> str:
        head = f"《{self.title}》" if self.title and not self.title.startswith("《") else self.title
        return f"{head}\n{self.content}".strip()

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "title": self.title, "content": self.content, "source": self.source}


class LawIndex:
    """文件式法条向量索引：构建、持久化、检索。"""

    def __init__(self, articles: List[LawArticle], vectors: np.ndarray, model: str = "") -> None:
        assert len(articles) == vectors.shape[0]
        self.articles = articles
        self.vectors = vectors.astype(np.float32)
        self.model = model

    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        corpus: Sequence[dict],
        embed_fn: EmbedFn,
        model: str = "",
        batch_size: int = 64,
        doc_id_field: str = "doc_id",
        title_field: str = "title",
        content_field: str = "content",
        source_field: str = "source",
    ) -> "LawIndex":
        """从语料构建索引。corpus 元素为 dict（字段名可由 data_loaders 归一化）。"""
        articles, texts = [], []
        for i, item in enumerate(corpus):
            art = LawArticle(
                doc_id=str(item.get(doc_id_field) or item.get("id") or f"law-{i:06d}"),
                title=str(item.get(title_field, "") or ""),
                content=str(item.get(content_field, "") or item.get("text", "")),
                source=str(item.get(source_field, "") or ""),
            )
            if not art.content:
                continue
            articles.append(art)
            texts.append(f"{art.title}。{art.content}" if art.title else art.content)

        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vec = embed_fn(batch)
            if vec.shape[0] != len(batch):
                raise ValueError(f"embedding 数量不匹配: {vec.shape[0]} != {len(batch)}")
            vectors.append(vec.astype(np.float32))
        all_vecs = np.concatenate(vectors, axis=0) if vectors else np.zeros((0, 1), np.float32)
        # L2 归一化（embed_fn 可能未归一化）
        norm = np.linalg.norm(all_vecs, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return cls(articles, all_vecs / norm, model=model)

    # ------------------------------------------------------------------ #
    def save(self, out_dir: str) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "count": len(self.articles),
            "dim": int(self.vectors.shape[1]),
            "model": self.model,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        np.save(out / VECTOR_NAME, self.vectors.astype(np.float16))
        with (out / METADATA_NAME).open("w", encoding="utf-8") as f:
            for art in self.articles:
                f.write(json.dumps(art.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, index_dir: str) -> "LawIndex":
        d = Path(index_dir)
        manifest = json.loads((d / MANIFEST_NAME).read_text(encoding="utf-8"))
        vectors = np.load(d / VECTOR_NAME).astype(np.float32)
        # 恢复 L2 归一化（float16 存储有微小误差）
        norm = np.linalg.norm(vectors, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vectors = vectors / norm
        articles: List[LawArticle] = []
        with (d / METADATA_NAME).open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                articles.append(LawArticle(**{k: item.get(k, "") for k in ("doc_id", "title", "content", "source")}))
        if len(articles) != vectors.shape[0]:
            raise ValueError(f"索引不一致: {len(articles)} 条元数据 vs {vectors.shape[0]} 向量")
        return cls(articles, vectors, model=str(manifest.get("model", "")))

    # ------------------------------------------------------------------ #
    def search_by_vector(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[LawArticle, float]]:
        """余弦相似度（内积）检索，返回 [(法条, 相似度)] 降序。"""
        if len(self.articles) == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-12)
        scores = (self.vectors @ q.T).reshape(-1)
        k = min(top_k, len(self.articles))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.articles[i], float(scores[i])) for i in idx]

    def search(self, query: str, embed_fn: EmbedFn, top_k: int = 10) -> List[Tuple[LawArticle, float]]:
        return self.search_by_vector(embed_fn([query])[0], top_k=top_k)

    def get_by_ids(self, doc_ids: Sequence[str]) -> List[LawArticle]:
        id_set = set(str(d) for d in doc_ids)
        return [a for a in self.articles if a.doc_id in id_set]


def build_or_load_index(cfg: AppConfig, corpus: Optional[Sequence[dict]], embed_fn: EmbedFn) -> LawIndex:
    """优先加载已有索引；否则用 corpus 现场构建并保存。"""
    index_dir = cfg.resolve_path(cfg.retrieval.index_dir)
    if (index_dir / MANIFEST_NAME).exists():
        return LawIndex.load(str(index_dir))
    if not corpus:
        raise FileNotFoundError(f"索引不存在且未提供语料: {index_dir}")
    index = LawIndex.build(corpus, embed_fn, model=cfg.embedding.model)
    index.save(str(index_dir))
    return index
