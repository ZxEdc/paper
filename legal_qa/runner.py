"""评测运行器：交互调试 / 批量多轮评测（论文第五章实验入口）。

用法：
  交互调试（需已启动 vLLM 与 embedding 服务）:
    python -m legal_qa.runner --interactive
  批量评测:
    python -m legal_qa.runner --dataset data/lecoqa.json --format lecoqa --limit 50 --concurrency 8
  STARD 检索质量评测（无需 LLM）:
    python -m legal_qa.evaluation --stard --corpus data/stard/corpus.json --queries data/stard/queries.json

输出（outputs/）:
  trace_*.jsonl  每轮: 用户消息/回答/调度建议/实际调用Skill/记忆卡片/时延
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import List, Optional

from .config import AppConfig
from .data_loaders import load_dialogues, load_law_corpus


async def run_dialogue_task(item: dict, cfg: AppConfig, sem: asyncio.Semaphore, top_k: int = 10) -> dict:
    """单个对话的评测任务（asyncio 并发受 sem 控制）。"""
    from .agent import LegalQAAssistant

    async with sem:
        assistant = LegalQAAssistant(cfg=cfg, corpus=None, user_id=str(item.get("id", "anon")))
        turns = item.get("turns") or ([item["question"]] if item.get("question") else [])
        records = []
        for i, user_text in enumerate(turns, start=1):
            result = await assistant.reply(user_text)
            records.append({
                "turn": i,
                "user": user_text,
                "assistant": result.content,
                "skill_hints": result.skill_hints,
                "skills_invoked": result.skills_invoked,
                "memory_card": result.memory_card,
                "latency": round(result.latency, 3),
            })
        return {
            "id": item.get("id"),
            "records": records,
            "final_memory": assistant.memory.mid.to_dict(),
            "trace": assistant.ctx.trace.records,
        }


async def run_dataset(
    cfg: AppConfig, dataset_path: str, out_path: str, limit: int = 0, concurrency: int = 8
) -> List[dict]:
    dialogues = load_dialogues(dataset_path)
    if limit:
        dialogues = dialogues[:limit]
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    results = []
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, item in enumerate(dialogues):
            r = await run_dialogue_task(item, cfg, sem)
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            done = i + 1
            if done % 5 == 0 or done == len(dialogues):
                print(f"[{done}/{len(dialogues)}] 已完成，累计 {time.time()-t0:.0f}s", flush=True)
    print(f"评测完成: {out}")
    return results


async def interactive(cfg: AppConfig) -> None:
    from .agent import LegalQAAssistant

    assistant = LegalQAAssistant(cfg=cfg)
    print("法律问答助手（输入 /quit 退出，/card 查看案件记忆，/trace 查看调度轨迹）")
    while True:
        try:
            user_text = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/card":
            print(assistant.memory.mid.to_card_text())
            continue
        if user_text == "/trace":
            for r in assistant.ctx.trace.records:
                print(r)
            continue
        result = await assistant.reply(user_text)
        print(f"\n助手> {result.content}")
        print(f"  [调度: {'->'.join(result.skill_hints)} | 实际: {','.join(result.skills_invoked) or '无'} | {result.latency:.1f}s]")


def main() -> None:
    parser = argparse.ArgumentParser(description="legal_qa 评测运行器")
    parser.add_argument("--interactive", action="store_true", help="交互调试模式")
    parser.add_argument("--dataset", type=str, help="数据集路径（json/jsonl）")
    parser.add_argument("--limit", type=int, default=0, help="限制评测条数")
    parser.add_argument("--concurrency", type=int, default=8, help="并发对话数")
    parser.add_argument("--out", type=str, default="", help="输出路径")
    args = parser.parse_args()

    cfg = AppConfig.from_env()
    if args.interactive:
        asyncio.run(interactive(cfg))
        return
    if not args.dataset:
        parser.error("需要 --dataset 或 --interactive")
    out = args.out or str(cfg.resolve_path(cfg.output_dir) / f"trace_{int(time.time())}.jsonl")
    asyncio.run(run_dataset(cfg, args.dataset, out, limit=args.limit, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
