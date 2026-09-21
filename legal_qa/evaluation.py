"""评测指标与 STARD 检索质量评测（论文第五章）。

指标：
  检索质量   Recall@K / MRR / nDCG@K（法条召回，对齐 STARD 官方）
  引用命中率 生成答案中引用的法条 doc_id 与 gold 的交集比例
  回答质量   ROUGE-L（自实现 LCS，无外部依赖）

STARD 评测不依赖 LLM，只需 embedding 服务：
  python -m legal_qa.evaluation --stard \
      --corpus data/stard/corpus.json --queries data/stard/queries.json \
      --topk 5 10 20
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from .config import AppConfig
from .data_loaders import cn_to_int, load_dialogues, load_law_corpus
from .llm import EmbeddingClient
from .retrieval import LawIndex, build_or_load_index


# ------------------------------------------------------------------ #
# 排序指标（gold 为相关集，二值相关）
# ------------------------------------------------------------------ #
def recall_at_k(ranked_ids: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for d in ranked_ids[:k] if d in gold)
    return hit / len(gold)


def mrr(ranked_ids: Sequence[str], gold: Set[str]) -> float:
    for rank, d in enumerate(ranked_ids, start=1):
        if d in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], gold: Set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(1.0 / np.log2(r + 1) for r, d in enumerate(ranked_ids[:k], start=1) if d in gold)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, min(len(gold), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ------------------------------------------------------------------ #
# 引用命中率：从生成文本中抽取 《法律名》第X条 引用
# ------------------------------------------------------------------ #
CITE_PATTERN = re.compile(r"《([^《》]{1,40}?)》[^。；]{0,12}?第([一二三四五六七八九十百千零两\d]+)条")


def extract_citations(text: str) -> List[str]:
    """抽取引用对 (法律名, 条序号) -> '法律名#条序号'（数字已归一化）。"""
    return [f"{name}#{cn_to_int(num)}" for name, num in CITE_PATTERN.findall(text)]


def citations_match(a: str, g: str) -> bool:
    """两条引用是否等价：条号相等且法名互为包含（'民法典#839' vs '中华人民共和国民法典#839'）。"""
    an, _, anum = a.partition("#")
    gn, _, gnum = g.partition("#")
    if anum != gnum or not anum:
        return False
    return an == gn or an in gn or gn in an


def citation_hit_rate(answers: Sequence[str], gold_texts: Sequence[str]) -> float:
    """引用命中率：答案抽取的引用（归一化）与参考答案抽取的引用有交集的比例。"""
    if not answers:
        return 0.0
    hits = 0
    for ans, ref in zip(answers, gold_texts):
        cites = extract_citations(ans)
        if not cites:
            continue
        gold_cites = set(extract_citations(ref))
        if gold_cites and any(c in gold_cites for c in cites):
            hits += 1
    return hits / len(answers)


# ------------------------------------------------------------------ #
# ROUGE-L（自实现）
# ------------------------------------------------------------------ #
def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            dp[j] = dp[j] if dp[j] >= prev and a[i - 1] != b[j - 1] else (
                dp[j - 1] if a[i - 1] == b[j - 1] else max(dp[j], prev)
            )
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            prev = tmp
    return dp[-1]


def rouge_l(prediction: str, reference: str, level: str = "char") -> float:
    """ROUGE-L F1。level: 'char' 按字（中文），'word' 按空格分词。"""
    p_tokens = list(prediction) if level == "char" else prediction.split()
    r_tokens = list(reference) if level == "char" else reference.split()
    if not p_tokens or not r_tokens:
        return 0.0
    lcs = _lcs_len(p_tokens, r_tokens)
    if lcs == 0:
        return 0.0
    prec, rec = lcs / len(p_tokens), lcs / len(r_tokens)
    return 2 * prec * rec / (prec + rec)


# ------------------------------------------------------------------ #
# STARD 检索质量评测（无 LLM）
# ------------------------------------------------------------------ #
def evaluate_stard(cfg: AppConfig, corpus_path: str, queries_path: str, topks: Iterable[int]) -> Dict:
    corpus = load_law_corpus(corpus_path)
    queries = load_dialogues(queries_path)
    embedder = EmbeddingClient(cfg.embedding)
    index = build_or_load_index(cfg, corpus, embedder.embed)

    ks = sorted(set(topks))
    metrics: Dict[str, List[float]] = {f"recall@{k}": [] for k in ks}
    metrics["mrr"] = []
    metrics["ndcg@10"] = []

    t0 = time.time()
    for i, q in enumerate(queries):
        gold = set(q["gold_ids"])
        if not gold:
            continue
        ranked = [a.doc_id for a, _ in index.search(q["question"], embedder.embed, top_k=max(ks))]
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
        metrics["mrr"].append(mrr(ranked, gold))
        metrics["ndcg@10"].append(ndcg_at_k(ranked, gold, 10))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(queries)} 已评测 ({time.time()-t0:.0f}s)", flush=True)

    summary = {name: round(float(np.mean(vals)), 4) for name, vals in metrics.items() if vals}
    summary["num_queries"] = len(metrics["mrr"])
    summary["corpus_size"] = len(corpus)
    return summary


# ------------------------------------------------------------------ #
# trace 结果的答案质量评测
# ------------------------------------------------------------------ #
def evaluate_traces(trace_path: str, dataset_path: str) -> Dict:
    """对 trace（baseline 或 runner 输出）计算答案质量（ROUGE-L + 引用命中率）。

    引用命中率优先使用数据集 gold 引用（match_name 解析），
    无 gold 引用时退化为与参考答案文本比对。
    """
    dialogues = {d["id"]: d for d in load_dialogues(dataset_path)}
    scores: List[float] = []
    cited = 0
    n_answers = 0
    with Path(trace_path).open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = dialogues.get(r.get("id"), {})
            ref = gold.get("answer", "")
            gold_cites = gold.get("gold_citations") or []
            for rec in r.get("records", []):
                ans = rec.get("assistant", "")
                if not ans:
                    continue
                n_answers += 1
                if ref:
                    scores.append(rouge_l(ans, ref))
                cites = extract_citations(ans)
                if gold_cites:
                    if any(citations_match(c, g) for c in cites for g in gold_cites):
                        cited += 1
                elif ref and citation_hit_rate([ans], [ref]) > 0:
                    cited += 1
    return {
        "rouge_l": round(float(np.mean(scores)), 4) if scores else None,
        "citation_hit_rate": round(cited / n_answers, 4) if n_answers else None,
        "num_answers": n_answers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="legal_qa 评测")
    parser.add_argument("--stard", action="store_true", help="STARD 检索质量评测")
    parser.add_argument("--corpus", type=str, default="", help="法条语料路径")
    parser.add_argument("--queries", type=str, default="", help="查询集路径")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--trace", type=str, default="", help="trace 文件（答案质量评测）")
    parser.add_argument("--dataset", type=str, default="", help="trace 对应的原始数据集")
    args = parser.parse_args()

    cfg = AppConfig.from_env()
    if args.stard:
        if not (args.corpus and args.queries):
            parser.error("--stard 需要 --corpus 和 --queries")
        summary = evaluate_stard(cfg, args.corpus, args.queries, args.topk)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        out = Path(cfg.resolve_path(cfg.output_dir))
        out.mkdir(parents=True, exist_ok=True)
        (out / f"stard_metrics_{int(time.time())}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    elif args.trace:
        if not args.dataset:
            parser.error("--trace 需要 --dataset")
        print(json.dumps(evaluate_traces(args.trace, args.dataset), ensure_ascii=False, indent=2))
    else:
        parser.error("需要 --stard 或 --trace")


if __name__ == "__main__":
    main()
