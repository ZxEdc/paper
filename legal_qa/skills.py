"""五个法律 Skill 与记忆驱动的动态调度器（论文创新点二）。

Skill 清单（对齐开题报告第四章）：
  1. CaseTypeSkill        案件类型识别
  2. FactExtractionSkill  关键事实抽取（含冲突检测）
  3. ClarificationSkill   追问生成（缺失槽位 / 冲突确认）
  4. LawRetrievalSkill    法规检索（记忆增强 RAG）
  5. AnswerReviewSkill    答案审查（引用核对 + 风险提示）

调度策略：
  dynamic —— 基于记忆状态的动态调用（本文方法）
  fixed   —— 固定顺序全量执行（消融实验对照组）

所有 Skill 只依赖 llm/retrieval/memory 模块，不依赖 agentscope；
agent.py 负责把它们注册为 AgentScope FunctionTool。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .llm import LLMClient
from .memory import HierarchicalMemory
from .retrieval import LawIndex

EmbedFn = Callable[[List[str]], "object"]  #返回 [N, dim] ndarray

CASE_TYPES = "劳动争议/婚姻家庭/借贷纠纷/侵权纠纷/合同纠纷/房产纠纷/继承纠纷/其他"


class SkillTrace:
    """Skill 调用轨迹（供论文分析调度行为）。"""

    def __init__(self) -> None:
        self.records: List[Dict] = []

    def log(self, skill: str, turn: int, detail: str = "", latency: float = 0.0) -> None:
        self.records.append({"turn": turn, "skill": skill, "detail": detail, "latency": round(latency, 3), "ts": time.strftime("%H:%M:%S")})


@dataclass
class SkillContext:
    """Skill 共享上下文：由 runner 每轮更新。"""
    memory: HierarchicalMemory
    llm: LLMClient
    law_index: Optional[LawIndex] = None
    embed_fn: Optional[EmbedFn] = None
    latest_user_msg: str = ""
    turn: int = 0
    trace: SkillTrace = field(default_factory=SkillTrace)
    retrieved_doc_ids: List[str] = field(default_factory=list)  # 本轮检索法条（引用核对用）


# ========================================================================= #
# Skill 1: 案件类型识别
# ========================================================================= #
class CaseTypeSkill:
    name = "identify_case_type"
    description = "识别用户咨询所属的案件类型（劳动争议/婚姻家庭/借贷纠纷/侵权纠纷/合同纠纷/房产纠纷/继承纠纷/其他）。每轮对话开始时如尚无案件类型应优先调用。"

    def __init__(self, ctx: SkillContext) -> None:
        self.ctx = ctx

    def run(self) -> str:
        m = self.ctx.memory.mid
        history = self.ctx.memory.short_window_text()
        prompt = (
            "你是法律咨询接待助手。根据对话判断案件类型，只输出 JSON："
            f'{{"case_type": "one of {CASE_TYPES}", "reason": "简短理由"}}\n\n'
            f"已知案件类型：{m.case_type or '未识别'}\n近期对话：\n{history[-1500:]}"
        )
        t0 = time.time()
        try:
            out = self.ctx.llm.chat_json([{"role": "user", "content": prompt}])
            ctype = str(out.get("case_type", "其他")).strip()
        except Exception as e:  # noqa: BLE001
            self.ctx.trace.log(self.name, self.ctx.turn, f"失败: {e}")
            return f"案件类型识别失败: {e}"
        if ctype and ctype != m.case_type:
            m.case_type = ctype
        self.ctx.trace.log(self.name, self.ctx.turn, f"case_type={ctype}", time.time() - t0)
        return f"案件类型已识别为「{ctype}」。"


# ========================================================================= #
# Skill 2: 关键事实抽取（含冲突检测）
# ========================================================================= #
class FactExtractionSkill:
    name = "extract_facts"
    description = "从用户最新消息中抽取关键事实（槽位-取值）、用户诉求与证据，写入中期案件记忆；同槽位出现矛盾取值时会触发冲突记录。每次收到用户消息后调用。"

    def __init__(self, ctx: SkillContext) -> None:
        self.ctx = ctx

    def run(self, user_text: str = "") -> str:
        ctx, m = self.ctx, self.ctx.memory.mid
        text = user_text or ctx.latest_user_msg
        slots = list(m.facts.keys())
        prompt = (
            "从法律咨询对话中抽取结构化信息，只输出 JSON：\n"
            '{"facts": [{"key": "槽位名", "value": "事实内容"}], '
            '"demands": ["用户诉求"], "evidence": ["证据"], "stage": "处理阶段判断"}\n'
            "槽位名尽量使用标准名（如：借款金额/交付方式/劳动关系起止时间/工资标准/损害后果等），"
            "语义相同的事实请合并，不要抽取对话中没有的信息。\n\n"
            f"案件类型：{m.case_type or '未知'}\n已有事实槽位：{slots}\n"
            f"用户最新消息：{text}\n近期对话：\n{ctx.memory.short_window_text()[-1200:]}"
        )
        t0 = time.time()
        try:
            out = ctx.llm.chat_json([{"role": "user", "content": prompt}])
        except Exception as e:  # noqa: BLE001
            ctx.trace.log(self.name, ctx.turn, f"失败: {e}")
            return f"事实抽取失败: {e}"
        conflicts: List[str] = []
        for f in out.get("facts", []):
            if isinstance(f, dict) and f.get("key") and f.get("value"):
                c = m.add_fact(str(f["key"]), str(f["value"]), ctx.turn, ctx.embed_fn)
                if c is not None:
                    conflicts.append(f"{c.key}:「{c.old_value}」vs「{c.new_value}」")
        for d in out.get("demands", []):
            if d and d not in m.demands:
                m.demands.append(str(d))
        for e in out.get("evidence", []):
            if e and e not in m.evidence:
                m.evidence.append(str(e))
        if out.get("stage"):
            m.stage = str(out["stage"])
        detail = f"+{len(out.get('facts', []))}条事实, 冲突{len(conflicts)}个"
        ctx.trace.log(self.name, ctx.turn, detail, time.time() - t0)
        if conflicts:
            return "事实已更新；检测到冲突待确认: " + "；".join(conflicts)
        return f"案件记忆已更新（完整度 {m.completeness():.0%}）。"


# ========================================================================= #
# Skill 3: 追问生成
# ========================================================================= #
class ClarificationSkill:
    name = "clarify"
    description = "当案件关键事实不完整或存在冲突时，生成一条自然、口语化的追问。事实完整且无冲突时返回'无需追问'。"

    def __init__(self, ctx: SkillContext) -> None:
        self.ctx = ctx

    def run(self) -> str:
        ctx, m = self.ctx, self.ctx.memory.mid
        # 优先处理未确认冲突
        conflicts = m.unresolved_conflicts()
        if conflicts:
            c = conflicts[0]
            question = (
                f"我想再跟您确认一下：您之前提到{c.key}是「{c.old_value}」，"
                f"刚才又说是「{c.new_value}」，请问以哪个为准呢？"
            )
            ctx.trace.log(self.name, ctx.turn, f"冲突确认: {c.key}")
            return question
        missing = m.missing_slots()
        if not missing:
            ctx.trace.log(self.name, ctx.turn, "无需追问")
            return "无需追问：关键事实已完整。"
        slot = missing[0]
        prompt = (
            f"你是耐心的法律咨询助手。案件类型「{m.case_type or '未知'}」，"
            f"还缺少关键信息「{slot}」。已知事实：\n{m.to_card_text()}\n"
            "请生成一句简短自然的中文追问（不要列举所有缺失项，只问这一个，口吻亲和）。只输出 JSON："
            '{"question": "追问内容"}'
        )
        try:
            out = ctx.llm.chat_json([{"role": "user", "content": prompt}])
            q = str(out.get("question", "")).strip()
        except Exception as e:  # noqa: BLE001
            ctx.trace.log(self.name, ctx.turn, f"失败: {e}")
            return f"请补充说明您的{slot}。"
        ctx.trace.log(self.name, ctx.turn, f"追问槽位: {slot}")
        return q or f"请补充说明您的{slot}。"


# ========================================================================= #
# Skill 4: 法规检索（记忆增强 RAG）
# ========================================================================= #
class LawRetrievalSkill:
    name = "search_laws"
    description = "基于案件记忆改写查询并检索相关法律条文，返回 Top-K 法条（含《法律名》第X条引用格式）。回答法律问题前应调用以获得依据。"

    def __init__(self, ctx: SkillContext, top_k: int = 10) -> None:
        self.ctx = ctx
        self.top_k = top_k

    def run(self, query: str = "") -> str:
        ctx = self.ctx
        if ctx.law_index is None or len(ctx.law_index.articles) == 0:
            ctx.trace.log(self.name, ctx.turn, "无索引")
            return "法规检索不可用（未加载法条索引）。"
        q = (query or ctx.latest_user_msg).strip()
        # ---- 记忆增强检索：拼接案件关键事实改写查询 ---- #
        facts = ctx.memory.mid.to_card_text()
        rewrite_prompt = (
            "把下面的法律咨询改写为适合法条检索的查询（保留争议焦点与关键事实，去掉口语），"
            '只输出 JSON：{"query": "改写后的查询"}\n'
            f"用户问题：{q}\n案件记忆：\n{facts[:800]}"
        )
        try:
            out = ctx.llm.chat_json([{"role": "user", "content": rewrite_prompt}])
            q = str(out.get("query", q)) or q
        except Exception:  # noqa: BLE001 - 改写失败退回原查询
            pass
        t0 = time.time()
        results = ctx.law_index.search(q, ctx.embed_fn, top_k=self.top_k)
        ctx.retrieved_doc_ids = [a.doc_id for a, _ in results]
        ctx.trace.log(self.name, ctx.turn, f"query={q[:30]}... 命中{len(results)}条", time.time() - t0)
        if not results:
            return "未检索到相关法条。"
        return "\n\n".join(f"[{i+1}] {a.to_prompt()}" for i, (a, s) in enumerate(results[:5]))


# ========================================================================= #
# Skill 5: 答案审查
# ========================================================================= #
class AnswerReviewSkill:
    name = "review_answer"
    description = "审查草拟的回答：核对法条引用是否来自本轮检索结果、是否包含必要风险提示、是否越界（如代替律师下结论）。返回审查意见与修订稿。给出最终回答前应调用。"

    def __init__(self, ctx: SkillContext) -> None:
        self.ctx = ctx

    def run(self, draft: str) -> str:
        ctx, m = self.ctx, self.ctx.memory.mid
        laws = ctx.law_index.get_by_ids(ctx.retrieved_doc_ids) if ctx.law_index else []
        law_texts = "\n".join(a.to_prompt() for a in laws[:5]) or "（本轮无检索结果）"
        prompt = (
            "你是法律公共服务质检员。审查下面的回答草稿，只输出 JSON：\n"
            '{"pass": true/false, "issues": ["问题列表"], "revised": "修订后的完整回答"}\n'
            "审查要点：1) 引用的法条必须来自【检索法条】且条文号正确；2) 存在风险标记时必须包含安全提示；"
            "3) 结尾应有'以上意见供参考，不构成正式法律意见'类声明；4) 不代替律师做确定性结论。\n\n"
            f"风险标记：{m.risk_flags or '无'}\n【检索法条】\n{law_texts[:2000]}\n\n"
            f"【回答草稿】\n{draft}"
        )
        t0 = time.time()
        try:
            out = ctx.llm.chat_json([{"role": "user", "content": prompt}])
            ok = bool(out.get("pass", True))
            issues = out.get("issues", [])
            revised = str(out.get("revised", draft)) or draft
        except Exception as e:  # noqa: BLE001
            ctx.trace.log(self.name, ctx.turn, f"失败: {e}")
            return draft  # 审查失败不阻塞回答
        ctx.trace.log(self.name, ctx.turn, f"pass={ok}, issues={len(issues)}", time.time() - t0)
        return revised


# ========================================================================= #
# Skill 动态调度器（创新点二核心）
# ========================================================================= #
@dataclass
class DispatchDecision:
    skill_hints: List[str]
    state_summary: str


class SkillDispatcher:
    """基于记忆状态的 Skill 调度。

    dynamic（本文方法）—— 规则驱动的状态机：
      轮次1 / 未识别类型   ->  identify_case_type
      有新用户消息          ->  extract_facts
      冲突未确认 / 缺槽位   ->  clarify
      需要法律依据          ->  search_laws
      出答案前              ->  review_answer
    fixed（消融对照）—— 每轮固定顺序全量执行。
    """

    def __init__(self, policy: str = "dynamic", completeness_threshold: float = 0.6) -> None:
        assert policy in ("dynamic", "fixed"), f"未知调度策略: {policy}"
        self.policy = policy
        self.threshold = completeness_threshold

    def dispatch(self, mem: HierarchicalMemory, turn: int, has_user_msg: bool = True) -> DispatchDecision:
        m = mem.mid
        if self.policy == "fixed":
            hints = ["identify_case_type", "extract_facts", "search_laws", "review_answer"]
            summary = "[固定流程模式] 每轮依次执行全部 Skill。"
            return DispatchDecision(hints, summary)
        # ---------------- 动态策略 ---------------- #
        hints: List[str] = []
        if not m.case_type:
            hints.append("identify_case_type")
        if has_user_msg:
            hints.append("extract_facts")
        if m.unresolved_conflicts() or m.completeness() < self.threshold:
            hints.append("clarify")
        hints.append("search_laws")
        hints.append("review_answer")
        summary = (
            f"[动态调度] 轮次{turn} | 案件类型: {m.case_type or '未识别'} | "
            f"事实完整度 {m.completeness():.0%}（阈值 {self.threshold:.0%}）| "
            f"未确认冲突 {len(m.unresolved_conflicts())} 个 | "
            f"建议执行: {' -> '.join(hints)}"
        )
        return DispatchDecision(hints, summary)


def render_dispatch_instruction(decision: DispatchDecision) -> str:
    """把调度决策渲染为注入本轮用户消息的执行指令。"""
    return (
        f"{decision.state_summary}\n"
        "请按上述建议调用工具完成本轮服务：先用 extract_facts 更新案件记忆，"
        "若建议包含 clarify 且确有缺失，应在回答末尾自然地追问；"
        "回答法律问题前用 search_laws 检索法条并在回答中引用；"
        "给出最终回答前用 review_answer 审查。"
    )
