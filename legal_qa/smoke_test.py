"""冒烟测试（无需 LLM/网络服务）：分层记忆 / 检索 / 调度 / 指标 / AgentScope 装配。

运行:  py -3.12 legal_qa/smoke_test.py   （或 python3 legal_qa/smoke_test.py）
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from legal_qa.config import AppConfig, LLMConfig  # noqa: E402
from legal_qa.llm import LLMClient  # noqa: E402
from legal_qa.memory import HierarchicalMemory, MidTermMemory  # noqa: E402
from legal_qa.retrieval import LawIndex  # noqa: E402
from legal_qa.skills import SkillContext, SkillDispatcher  # noqa: E402
from legal_qa.evaluation import extract_citations, mrr, ndcg_at_k, recall_at_k, rouge_l  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name if not detail else f"{name} ({detail})")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))


# ---------------------------------------------------------------- #
def fake_embed(texts):
    """确定性伪向量：按字符哈希，保证可复现。"""
    out = []
    for t in texts:
        v = np.zeros(32, dtype=np.float32)
        for ch in t[:16]:
            v[ord(ch) % 32] += 1.0
        n = np.linalg.norm(v)
        out.append(v / (n if n > 0 else 1.0))
    return np.stack(out)


def test_memory() -> None:
    print("\n[1] 分层记忆")
    mem = HierarchicalMemory(user_id="t1", window=3, embed_fn=fake_embed)
    m = mem.mid
    m.case_type = "借贷纠纷"
    m.add_fact("借款金额", "3万元", 1, fake_embed)
    m.add_fact("交付方式", "微信转账", 1, fake_embed)
    m.add_fact("有无借据", "有借条", 2, fake_embed)
    c = m.add_fact("借款金额", "5万元", 3, fake_embed)  # 冲突
    check("冲突检测", c is not None and c.key == "借款金额")
    check("完整度", 0.0 < m.completeness() < 1.0, f"got {m.completeness()}")
    check("缺失槽位", "约定利息" in m.missing_slots() and "还款情况" in m.missing_slots())
    check("冲突确认后固化", m.resolve_conflict("借款金额", "5万元") and m.facts["借款金额"].value == "5万元")
    recalled = m.recall("借了多少钱", fake_embed, top_k=3)
    check("记忆召回", any(k == "借款金额" for k, _ in recalled))
    m.scan_risk("对方威胁我")
    check("风险扫描", "威胁" in m.risk_flags)
    mem.append_turn("user", "你好")
    mem.append_turn("assistant", "您好，请讲")
    text = mem.render_prompt_sections("借钱问题")
    check("prompt 记忆区", all(k in text for k in ("案件记忆", "短期记忆")))
    with tempfile.TemporaryDirectory() as td:
        mem.save(td)
        mem2 = HierarchicalMemory(user_id="t1")
        mem2.load(td)
        check("记忆持久化", mem2.mid.case_type == "借贷纠纷" and "借款金额" in mem2.mid.facts)


def test_retrieval() -> None:
    print("\n[2] 法条检索索引")
    corpus = [
        {"doc_id": "A1", "title": "民法典", "content": "借款合同应当采用书面形式，但是自然人之间借款另有约定的除外。"},
        {"doc_id": "A2", "title": "劳动法", "content": "工资应当以货币形式按月支付给劳动者本人。不得克扣或者无故拖欠。"},
        {"doc_id": "A3", "title": "民法典", "content": "向人民法院请求保护民事权利的诉讼时效期间为三年。"},
    ]
    idx = LawIndex.build(corpus, fake_embed, model="fake")
    with tempfile.TemporaryDirectory() as td:
        idx.save(td)
        idx2 = LawIndex.load(td)
        res = idx2.search("拖欠工资怎么办", fake_embed, top_k=3)
        check("检索返回", len(res) == 3 and res[0][1] >= res[-1][1])
        check("语义命中Top1", res[0][0].doc_id == "A2", f"got {res[0][0].doc_id}")
        check("按ID取法条", len(idx2.get_by_ids(["A1", "A3"])) == 2)


def test_dispatcher() -> None:
    print("\n[3] Skill 动态调度")
    mem = HierarchicalMemory(user_id="t2", embed_fn=None)
    d = SkillDispatcher(policy="dynamic")
    dec = d.dispatch(mem, 1)
    check("首轮含类型识别", "identify_case_type" in dec.skill_hints and "clarify" in dec.skill_hints)
    mem.mid.case_type = "劳动争议"
    for k in ["劳动关系起止时间", "工资标准", "欠薪金额或时长", "是否签劳动合同", "离职原因"]:
        mem.mid.add_fact(k, "x", 1)
    dec2 = d.dispatch(mem, 5)
    check("完整后不追问", "clarify" not in dec2.skill_hints, str(dec2.skill_hints))
    df = SkillDispatcher(policy="fixed")
    decf = df.dispatch(mem, 3)
    check("固定流程对照组", decf.skill_hints == ["identify_case_type", "extract_facts", "search_laws", "review_answer"])
    check("状态摘要", "事实完整度" in dec.state_summary)


def test_skill_fallback() -> None:
    print("\n[4] Skill 降级（无 LLM 服务时不崩溃）")
    ctx = SkillContext(
        memory=HierarchicalMemory(user_id="t3"),
        llm=LLMClient(LLMConfig(api_base_url="http://127.0.0.1:1/v1")),
    )
    ctx.latest_user_msg = "公司欠我三个月工资"
    from legal_qa.skills import CaseTypeSkill, ClarificationSkill, FactExtractionSkill

    r1 = CaseTypeSkill(ctx).run()
    check("类型识别降级", "识别失败" in r1)
    r2 = FactExtractionSkill(ctx).run()
    check("事实抽取降级", "抽取失败" in r2)
    ctx.memory.mid.case_type = "劳动争议"
    ctx.memory.mid.conflicts.append(type(ctx.memory.mid.conflicts[0]) if ctx.memory.mid.conflicts else
                                    __import__("legal_qa.memory", fromlist=["ConflictRecord"]).ConflictRecord(
                                        "工资标准", "3千", "5千", turn=1))
    r3 = ClarificationSkill(ctx).run()
    check("冲突追问(无LLM路径)", "确认" in r3)


def test_metrics() -> None:
    print("\n[5] 评测指标")
    gold = {"A1", "A2"}
    ranked = ["A3", "A1", "A2", "A4"]
    check("recall@2", recall_at_k(ranked, gold, 2) == 0.5)
    check("recall@3", recall_at_k(ranked, gold, 3) == 1.0)
    check("mrr", abs(mrr(ranked, gold) - 0.5) < 1e-9)
    check("ndcg", 0.0 < ndcg_at_k(ranked, gold, 3) < 1.0)
    check("rouge_l完全一致", rouge_l("劳动合同", "劳动合同") == 1.0)
    check("rouge_l部分", 0.0 < rouge_l("劳动合同纠纷", "劳动争议") < 1.0)
    cites = extract_citations("依据《民法典》第六百七十五条，借款人应当按照约定的期限返还借款。")
    check("引用抽取", cites == ["民法典#675"])


def test_agent_wiring() -> None:
    print("\n[6] AgentScope 装配（无网络，仅构建）")
    try:
        from legal_qa.agent import LegalQAAssistant

        cfg = AppConfig()
        cfg.llm.api_base_url = "http://127.0.0.1:1/v1"
        a = LegalQAAssistant(cfg=cfg, corpus=None, user_id="smoke")
        fns = a._build_tool_functions()
        names = [f.__name__ for f in fns]
        check("5个Skill工具", names == ["identify_case_type", "extract_facts", "clarify", "search_laws", "review_answer"])
        import asyncio

        async def build():
            return await a._build_agent()

        try:
            agent = asyncio.run(build())
            check("Agent构建", agent is not None)
        except Exception as e:  # noqa: BLE001
            check("Agent构建", False, f"{type(e).__name__}: {e}")
    except ImportError as e:
        check("agentscope 可导入", False, str(e))


def main() -> None:
    print("=" * 56)
    print("legal_qa 冒烟测试（无 LLM / 无网络）")
    print("=" * 56)
    test_memory()
    test_retrieval()
    test_dispatcher()
    test_skill_fallback()
    test_metrics()
    test_agent_wiring()
    print("\n" + "=" * 56)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项:", "; ".join(FAIL))
        sys.exit(1)
    print("全部通过 ✔")


if __name__ == "__main__":
    main()
