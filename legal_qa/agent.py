"""AgentScope 组装层：把分层记忆、五个 Skill、RAG 装配为法律问答智能体。

适配 agentscope 2.x API：
  Agent / ReActConfig / OpenAIChatModel / OpenAICredential / FunctionTool / Toolkit / UserMsg

每轮对话的消息封套（envelope）结构：
  【分层记忆区】(长期画像 + 中期案件卡片 + 短期窗口)  <- 创新点一
  【动态调度指令】(状态摘要 + Skill 建议序列)          <- 创新点二
  【用户消息】
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .config import AppConfig
from .llm import EmbeddingClient, LLMClient
from .memory import HierarchicalMemory
from .retrieval import LawIndex, build_or_load_index
from .skills import (
    AnswerReviewSkill,
    CaseTypeSkill,
    ClarificationSkill,
    FactExtractionSkill,
    LawRetrievalSkill,
    SkillContext,
    SkillDispatcher,
    SkillTrace,
    render_dispatch_instruction,
)

SYSTEM_PROMPT = """你是"法律公共服务问答系统"的咨询助手，面向普通公众提供免费法律咨询引导。

职责与风格：
1. 用通俗易懂的语言解释法律问题，必要时引用《法律名》第X条作为依据；
2. 耐心引导用户补全关键事实，一次只追问一个最重要的问题；
3. 发现用户面临人身安全等风险时，优先给出安全提示与求助渠道（如报警、妇联、法律援助热线12348）；
4. 不代替律师下确定性结论，重大事项建议用户线下咨询律师或申请法律援助；
5. 每个完整回答的结尾附一句"以上意见供参考，不构成正式法律意见"。

你可以调用以下工具完成服务：识别案件类型、抽取案件事实、生成追问、检索法条、审查回答。请根据每轮记忆状态与调度指令决定调用顺序。"""

MAX_REACT_ITERS = 8


@dataclass
class TurnResult:
    """单轮结果（含论文分析所需的轨迹信息）。"""
    content: str
    turn: int
    skill_hints: List[str]
    memory_card: str
    latency: float
    skills_invoked: List[str] = field(default_factory=list)


class LegalQAAssistant:
    """法律问答智能体工厂与运行时。"""

    def __init__(
        self,
        cfg: Optional[AppConfig] = None,
        corpus: Optional[Sequence[dict]] = None,
        user_id: str = "default",
    ) -> None:
        self.cfg = cfg or AppConfig.from_env()
        self.llm = LLMClient(self.cfg.llm)
        self.embedder = EmbeddingClient(self.cfg.embedding)
        self.corpus = list(corpus) if corpus else None
        self.user_id = user_id
        self.memory = HierarchicalMemory(user_id=user_id, window=6, embed_fn=self.embedder.embed)
        self.dispatcher = SkillDispatcher(
            policy=self.cfg.dispatch.policy,
            completeness_threshold=self.cfg.dispatch.fact_completeness_threshold,
        )
        self.ctx = SkillContext(memory=self.memory, llm=self.llm, embed_fn=self.embedder.embed)
        self._skills = {
            "case_type": CaseTypeSkill(self.ctx),
            "fact_extraction": FactExtractionSkill(self.ctx),
            "clarification": ClarificationSkill(self.ctx),
            "law_retrieval": None,   # 延迟构建（需要索引）
            "answer_review": AnswerReviewSkill(self.ctx),
        }
        self._agent = None

    # ------------------------------------------------------------------ #
    def _ensure_index(self) -> Optional[LawIndex]:
        if self._skills["law_retrieval"] is None:
            try:
                index = build_or_load_index(self.cfg, self.corpus, self.embedder.embed)
                self.ctx.law_index = index
                self._skills["law_retrieval"] = LawRetrievalSkill(self.ctx, top_k=self.cfg.retrieval.top_k)
            except FileNotFoundError:
                self.ctx.law_index = None  # 检索不可用时降级运行
        return self.ctx.law_index

    # ------------------------------------------------------------------ #
    def _build_tool_functions(self):
        """把 5 个 Skill 包装为可注册的普通函数（闭包共享 SkillContext）。"""
        ctx = self.ctx

        def identify_case_type() -> str:
            """识别用户咨询所属的案件类型（劳动争议/婚姻家庭/借贷纠纷/侵权纠纷/合同纠纷/房产纠纷/继承纠纷/其他）。"""
            return self._skills["case_type"].run()

        def extract_facts(user_text: str = "") -> str:
            """从用户最新消息抽取关键事实、诉求与证据并写入案件记忆；同槽位矛盾取值会记录冲突。"""
            return self._skills["fact_extraction"].run(user_text)

        def clarify() -> str:
            """关键事实不完整或存在冲突时，生成一条自然的追问；事实完整时返回无需追问。"""
            return self._skills["clarification"].run()

        def search_laws(query: str = "") -> str:
            """基于案件记忆改写查询并检索相关法律条文，返回 Top-K 法条。回答法律问题前应调用。"""
            skill = self._skills["law_retrieval"]
            if skill is None:
                return "法规检索不可用（未加载法条索引）。"
            return skill.run(query)

        def review_answer(draft: str) -> str:
            """审查回答草稿（法条引用核对/风险提示/边界声明），返回修订稿。给出最终回答前应调用。"""
            return self._skills["answer_review"].run(draft)

        return [identify_case_type, extract_facts, clarify, search_laws, review_answer]

    async def _build_agent(self):
        """构建 AgentScope Agent（agentscope 2.x API）。"""
        from agentscope.agent import Agent, ReActConfig
        from agentscope.credential import OpenAICredential
        from agentscope.model import OpenAIChatModel
        from agentscope.tool import FunctionTool, Toolkit

        self._ensure_index()
        toolkit = Toolkit()
        for fn in self._build_tool_functions():
            await toolkit.add_tool(
                FunctionTool(fn, name=fn.__name__, description=(fn.__doc__ or "").strip().splitlines()[0])
            )
        credential = OpenAICredential(
            api_key=self.cfg.llm.api_key or "EMPTY",
            base_url=self.cfg.llm.api_base_url,
        )
        model = OpenAIChatModel(credential=credential, model=self.cfg.llm.model)
        return Agent(
            name="legal_qa_assistant",
            system_prompt=SYSTEM_PROMPT,
            model=model,
            toolkit=toolkit,
            react_config=ReActConfig(max_iters=MAX_REACT_ITERS),
        )

    async def ensure_agent(self):
        if self._agent is None:
            self._agent = await self._build_agent()
        return self._agent

    # ------------------------------------------------------------------ #
    async def reply(self, user_text: str) -> TurnResult:
        """单轮对话：组装记忆封套 -> ReAct 推理 -> 记忆更新。"""
        from agentscope.message import UserMsg

        agent = await self.ensure_agent()
        turn = self.ctx.turn + 1
        self.ctx.turn = turn
        self.ctx.latest_user_msg = user_text
        self.ctx.retrieved_doc_ids = []

        decision = self.dispatcher.dispatch(self.memory, turn)
        before_trace = set(r["skill"] for r in self.ctx.trace.records)
        envelope = (
            f"{self.memory.render_prompt_sections(user_text, self.cfg.dispatch.memory_recall_top_k)}\n\n"
            f"{render_dispatch_instruction(decision)}\n\n"
            f"【用户消息】{user_text}"
        )
        t0 = time.time()
        try:
            reply_msg = await agent.reply(UserMsg("user", envelope))
            content = reply_msg.get_text_content() or ""
        except Exception as e:  # noqa: BLE001 - 模型不可用时给出可诊断的错误
            content = f"[系统错误] {type(e).__name__}: {e}"
        latency = time.time() - t0

        # 短期记忆更新
        self.memory.append_turn("user", user_text)
        self.memory.append_turn("assistant", content)
        # 兜底：智能体未主动抽取事实时，程序化更新中期记忆（保证下一轮调度状态正确）
        if "extract_facts" not in (set(r["skill"] for r in self.ctx.trace.records) - before_trace):
            try:
                self._skills["fact_extraction"].run(user_text)
            except Exception:  # noqa: BLE001
                pass

        invoked = sorted(set(r["skill"] for r in self.ctx.trace.records) - before_trace)
        return TurnResult(
            content=content,
            turn=turn,
            skill_hints=decision.skill_hints,
            memory_card=self.memory.mid.to_card_text(),
            latency=latency,
            skills_invoked=invoked,
        )

    # ------------------------------------------------------------------ #
    async def run_dialogue(self, turns: Sequence[str]) -> List[TurnResult]:
        """多轮对话（LeCoDe/LeCoQA 评测入口）。"""
        results = []
        for t in turns:
            results.append(await self.reply(t))
            if len(results) >= self.cfg.dispatch.max_dialogue_turns:
                break
        return results
