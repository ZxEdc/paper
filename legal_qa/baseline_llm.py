"""基线实验 1：纯 LLM 直接问答（无记忆 / 无检索 / 无 Skill 调度）。

作为论文主对比实验的下限基线：模型只拿到系统提示词 + 对话原文，
不使用任何工具、检索或记忆机制；多轮输入仅做朴素的对话历史拼接。

本地即可运行（无需 GPU），支持任何 OpenAI 兼容 API（DeepSeek / Qwen / GLM...）。

用法：
  # 1. 配置 DeepSeek（也可写进 legal_qa/.env）
  set LLM_API_BASE_URL=https://api.deepseek.com/v1
  set LLM_MODEL=deepseek-chat
  set LLM_API_KEY=sk-xxxxxx

  # 2. 跑基线（先用 demo 数据试通，再换 LeCoQA）
  py -3.12 -m legal_qa.baseline_llm --dataset legal_qa/data/demo_lecoqa.json
  py -3.12 -m legal_qa.baseline_llm --dataset data/lecoqa.json --limit 200 --concurrency 8

  # 3. 评测（ROUGE-L + 引用命中率）
  py -3.12 -m legal_qa.evaluation --trace outputs/baseline_llm_*.jsonl --dataset data/lecoqa.json
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from .config import AppConfig
from .data_loaders import load_dialogues
from .llm import LLMClient

BASELINE_SYSTEM_PROMPT = """你是法律公共服务咨询助手，面向普通公众提供免费法律咨询引导。
用通俗易懂的中文回答，回答应给出法律依据（《法律名》第X条）；
发现人身安全风险时优先给出安全提示与求助渠道；
重大事项建议用户线下咨询律师或申请法律援助（12348）；
每个完整回答的结尾附一句"以上意见供参考，不构成正式法律意见"。
（注意：本基线不使用任何检索、记忆或工具机制，仅凭模型自身知识作答。）"""


def run_one(llm: LLMClient, item: Dict, max_turns: int) -> Dict:
    """单条对话：朴素拼接对话历史，无任何记忆/检索机制。"""
    messages = [{"role": "system", "content": BASELINE_SYSTEM_PROMPT}]
    records: List[Dict] = []
    for i, turn in enumerate(item["turns"][:max_turns], start=1):
        messages.append({"role": "user", "content": turn})
        t0 = time.time()
        try:
            answer = llm.chat(messages)
            err = ""
        except Exception as e:  # noqa: BLE001 - 单条失败不中断整体评测
            answer, err = "", f"{type(e).__name__}: {e}"
        messages.append({"role": "assistant", "content": answer or "(调用失败)"})
        records.append({
            "turn": i,
            "user": turn,
            "assistant": answer,
            "error": err,
            "latency": round(time.time() - t0, 3),
        })
        if not answer:
            break
    return {"id": item["id"], "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="基线1：纯 LLM 直接问答（OpenAI 兼容 API）")
    parser.add_argument("--dataset", required=True, help="数据集路径（json/jsonl）")
    parser.add_argument("--out", default="", help="输出 trace 路径（默认 outputs/baseline_llm_<时间戳>.jsonl）")
    parser.add_argument("--limit", type=int, default=0, help="限制评测条数（先小规模试跑）")
    parser.add_argument("--concurrency", type=int, default=8, help="并发请求数")
    parser.add_argument("--max-turns", type=int, default=30, help="多轮对话轮数上限")
    args = parser.parse_args()

    cfg = AppConfig.from_env()
    print(f"模型: {cfg.llm.model} @ {cfg.llm.api_base_url}")
    if cfg.llm.api_key in ("", "EMPTY"):
        print("警告: 未检测到 API key，请设置 LLM_API_KEY（或写入 legal_qa/.env）")

    dialogues = load_dialogues(args.dataset)
    if args.limit:
        dialogues = dialogues[: args.limit]
    print(f"加载 {len(dialogues)} 条对话")

    out_path = Path(args.out) if args.out else (
        cfg.resolve_path(cfg.output_dir) / f"baseline_llm_{int(time.time())}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLMClient(cfg.llm)
    t0 = time.time()
    done, failed = 0, 0
    with out_path.open("w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_one, llm, item, args.max_turns): item for item in dialogues}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    result = fut.result()
                    if any(r.get("error") for r in result["records"]):
                        failed += 1
                except Exception as e:  # noqa: BLE001
                    result = {"id": item["id"], "records": [], "fatal": f"{type(e).__name__}: {e}"}
                    failed += 1
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                done += 1
                if done % 10 == 0 or done == len(dialogues):
                    print(f"[{done}/{len(dialogues)}] 失败{failed} 累计{time.time()-t0:.0f}s", flush=True)

    print(f"\n完成: {out_path}（失败 {failed} 条）")
    print(f"评测: py -m legal_qa.evaluation --trace {out_path} --dataset {args.dataset}")


if __name__ == "__main__":
    main()
