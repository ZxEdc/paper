"""数据加载器：STARD / LeCoQA / LeCoDe（字段名宽松映射）。

各数据集官方仓库字段可能随版本调整，这里统一归一化为：
  对话条目: {"id": str, "question": str, "turns": [str], "answer": str, "gold_ids": [str]}
  法条语料: [{"doc_id": str, "title": str, "content": str, "source": str}]

支持 json / jsonl；未知字段尝试常见别名（query/caption/question、
articles/laws/relevant_laws/gold 等）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

QUESTION_KEYS = ["question", "query", "caption", "q", "text", "问题"]
ANSWER_KEYS = ["answer", "response", "reply", "gold_answer", "答案文本", "参考答案"]
GOLD_KEYS = ["gold_ids", "relevant_laws", "gold_articles", "laws", "articles", "law_ids", "cited_articles", "labels", "match_id"]
MATCH_NAME_KEYS = ["match_name", "相关法规"]
TURNS_KEYS = ["turns", "utterances", "dialogue", "messages", "questions"]
ID_KEYS = ["id", "query_id", "case_id", "qid", "no"]

CONTENT_KEYS = ["content", "text", "article", "body", "law_content", "provision"]
TITLE_KEYS = ["title", "name", "law_name", "article_title"]
DOCID_KEYS = ["doc_id", "id", "article_id", "law_id", "index", "no"]

# ---------------- 中文数字转换（法条条号归一化） ---------------- #
_CN_NUM = {"零": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNIT = {"十": 10, "百": 100, "千": 1000}
_LAW_REF_PATTERN = re.compile(r"^(.+?)第([一二三四五六七八九十百千零两\d]+)条$")


def cn_to_int(s: str) -> int:
    """中文数字/阿拉伯数字 -> int（第一千零七十九 -> 1079）。"""
    if s.isdigit():
        return int(s)
    total, num = 0, 0
    for ch in s:
        if ch in _CN_NUM:
            num = _CN_NUM[ch]
        elif ch in _CN_UNIT:
            unit = _CN_UNIT[ch]
            total += (num or 1) * unit
            num = 0
    return total + num


def parse_law_ref(ref: str) -> Optional[Tuple[str, int]]:
    """解析 '中华人民共和国民法典第八百三十九条' -> ('中华人民共和国民法典', 839)。"""
    m = _LAW_REF_PATTERN.match(str(ref).strip())
    if not m:
        return None
    try:
        return m.group(1), cn_to_int(m.group(2))
    except Exception:  # noqa: BLE001
        return None


def _first(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default


def _read_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        data = json.loads(stripped)
        if isinstance(data, dict):
            # 常见包装：{"data": [...]} / {"queries": [...]} / {"items": [...]}
            for k in ("data", "queries", "items", "samples", "dialogs", "dialogues", "examples"):
                if isinstance(data.get(k), list):
                    return data[k]
            return [data]
        return data
    # jsonl
    records = []
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _norm_gold(raw: Any) -> List[str]:
    """gold 法条引用归一化为 doc_id 列表。支持 str/list[dict]/list[str]。"""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    out: List[str] = []
    for item in raw if isinstance(raw, list) else [raw]:
        if isinstance(item, dict):
            out.append(str(_first(item, DOCID_KEYS, "")))
        elif item is not None:
            out.append(str(item))
    return [g for g in out if g]


def load_dialogues(path: str) -> List[Dict[str, Any]]:
    """加载对话数据集（STARD 查询 / LeCoQA / LeCoDe 多轮均可）。"""
    records = _read_records(path)
    dialogues: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        question = _first(rec, QUESTION_KEYS, "")
        turns_raw = _first(rec, TURNS_KEYS, None)
        turns: List[str] = []
        if isinstance(turns_raw, list) and turns_raw:
            for t in turns_raw:
                if isinstance(t, str):
                    turns.append(t)
                elif isinstance(t, dict):
                    turns.append(str(_first(t, QUESTION_KEYS + ["content", "utterance"], "")))
            turns = [t for t in turns if t]
        if not turns and question:
            turns = [str(question)]
        if not turns:
            continue
        # gold 引用（从 match_name / 相关法规 键名解析出 法名#条号）
        gold_citations: List[str] = []
        names = rec.get("match_name") or []
        if isinstance(names, str):
            names = [names]
        laws_dict = rec.get("相关法规")
        if isinstance(laws_dict, dict):
            names = names + list(laws_dict.keys())
        for nm in names:
            parsed = parse_law_ref(nm)
            if parsed:
                gold_citations.append(f"{parsed[0]}#{parsed[1]}")
        dialogues.append({
            "id": str(_first(rec, ID_KEYS, f"case-{i:05d}")),
            "question": str(question or turns[0]),
            "turns": turns,
            "answer": str(_first(rec, ANSWER_KEYS, "") or ""),
            "gold_ids": _norm_gold(_first(rec, GOLD_KEYS, None)),
            "gold_citations": gold_citations,
        })
    if not dialogues:
        raise ValueError(f"未能从 {path} 解析出任何对话条目，请检查字段名")
    return dialogues


def load_law_corpus(path: str) -> List[Dict[str, Any]]:
    """加载法条语料（STARD 55k 法条等）。"""
    records = _read_records(path)
    corpus: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        if isinstance(rec, str):
            corpus.append({"doc_id": f"law-{i:06d}", "title": "", "content": rec, "source": ""})
            continue
        if not isinstance(rec, dict):
            continue
        doc_id = _first(rec, DOCID_KEYS, None)
        content = _first(rec, CONTENT_KEYS, "")
        if content in (None, ""):
            continue
        corpus.append({
            "doc_id": str(doc_id if doc_id is not None else f"law-{i:06d}"),
            "title": str(_first(rec, TITLE_KEYS, "") or ""),
            "content": str(content),
            "source": str(_first(rec, ["source", "from", "origin"], "") or ""),
        })
    if not corpus:
        raise ValueError(f"未能从 {path} 解析出法条语料，请检查字段名")
    return corpus
