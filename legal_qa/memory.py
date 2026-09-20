"""分层记忆（论文创新点一）。

三层结构（对齐开题报告第三章）：
  短期记忆  当前会话的最近对话窗口（原文保留，供 prompt 直接拼接）
  中期记忆  案件事实结构化卡片：案件类型 / 关键事实 / 证据 / 处理阶段 / 用户诉求 / 风险标记
  长期记忆  用户画像：地域、身份、偏好等跨会话稳定属性

记忆表示 = 文本 + 结构化字段 + 向量：
  - 中期记忆每条事实同时保存 embedding，支持基于相似度的记忆召回
  - 冲突检测：同槽位出现不同取值时生成 ConflictRecord，
    由追问 Skill 向用户求证后按 confirm 结果固化

本模块不依赖 agentscope，可独立单测（向量注入 embed_fn 即可）。
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

EmbedFn = Callable[[List[str]], np.ndarray]

# 各案件类型的关键事实槽位（追问 Skill 用于判断事实完整性）
CASE_TYPE_SLOTS: Dict[str, List[str]] = {
    "劳动争议": ["劳动关系起止时间", "工资标准", "欠薪金额或时长", "是否签劳动合同", "离职原因"],
    "婚姻家庭": ["婚姻存续时间", "主要争议财产", "子女情况", "对方态度"],
    "借贷纠纷": ["借款金额", "交付方式", "约定利息", "还款情况", "有无借据"],
    "侵权纠纷": ["损害发生时间", "损害后果", "因果关系", "过错方", "损失金额"],
    "合同纠纷": ["合同类型", "违约行为", "损失金额", "催告情况"],
    "房产纠纷": ["房产性质", "权属登记情况", "争议焦点", "占款金额"],
    "继承纠纷": ["被继承人死亡时间", "遗产范围", "继承人数", "有无遗嘱"],
    "其他": ["争议事项", "时间", "损失"],
}

RISK_KEYWORDS = ["暴力", "威胁", "自杀", "自残", "报复", "伤害他人"]


@dataclass
class Fact:
    """中期记忆中的一条关键事实（结构化字段 + 向量）。"""
    key: str                      # 槽位名，如 "借款金额"
    value: str                    # 事实内容，如 "3万元，2024年3月现金交付"
    source_turn: int = 0          # 来源轮次
    confidence: float = 1.0
    fact_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    embedding: Optional[List[float]] = None

    def text(self) -> str:
        return f"{self.key}: {self.value}"


@dataclass
class ConflictRecord:
    """事实冲突记录（冲突检测与确认更新机制的载体）。"""
    key: str
    old_value: str
    new_value: str
    resolved: bool = False
    confirmed_value: Optional[str] = None
    turn: int = 0


class MidTermMemory:
    """中期记忆：案件事实卡片。"""

    def __init__(self) -> None:
        self.case_type: str = ""            # 案件类型（如 劳动争议）
        self.facts: Dict[str, Fact] = {}    # slot -> Fact
        self.evidence: List[str] = []       # 用户提及的证据材料
        self.stage: str = "咨询初期"        # 处理阶段
        self.demands: List[str] = []        # 用户诉求
        self.risk_flags: List[str] = []     # 风险标记
        self.conflicts: List[ConflictRecord] = []

    # ---------------- 事实增改 + 冲突检测 ---------------- #
    def add_fact(self, key: str, value: str, turn: int = 0, embed_fn: Optional[EmbedFn] = None) -> Optional[ConflictRecord]:
        key, value = key.strip(), value.strip()
        if not key or not value:
            return None
        emb: Optional[List[float]] = None
        if embed_fn is not None:
            try:
                emb = embed_fn([f"{key}: {value}"])[0].tolist()
            except Exception:  # noqa: BLE001 - embedding 失败不阻塞记忆更新
                emb = None
        old = self.facts.get(key)
        if old is None:
            self.facts[key] = Fact(key, value, turn, embedding=emb)
            return None
        if old.value == value:
            return None
        # 同槽位不同取值 -> 冲突，待确认
        conflict = ConflictRecord(key, old.value, value, turn=turn)
        self.conflicts.append(conflict)
        if emb is not None:
            old.embedding = emb  # 先暂存新向量，确认后写入新值
        return conflict

    def resolve_conflict(self, key: str, confirmed_value: str) -> bool:
        """用户确认后固化事实。"""
        if key not in self.facts:
            return False
        self.facts[key].value = confirmed_value
        for c in self.conflicts:
            if c.key == key and not c.resolved:
                c.resolved, c.confirmed_value = True, confirmed_value
        return True

    # ---------------- 状态查询 ---------------- #
    def completeness(self) -> float:
        """事实完整度 = 已填槽位 / 该案件类型必填槽位。"""
        slots = CASE_TYPE_SLOTS.get(self.case_type or "其他", CASE_TYPE_SLOTS["其他"])
        filled = sum(1 for s in slots if self.facts.get(s) and self.facts[s].value)
        return filled / len(slots) if slots else 1.0

    def missing_slots(self) -> List[str]:
        slots = CASE_TYPE_SLOTS.get(self.case_type or "其他", CASE_TYPE_SLOTS["其他"])
        return [s for s in slots if not (self.facts.get(s) and self.facts[s].value)]

    def unresolved_conflicts(self) -> List[ConflictRecord]:
        return [c for c in self.conflicts if not c.resolved]

    def scan_risk(self, text: str) -> None:
        for kw in RISK_KEYWORDS:
            if kw in text and kw not in self.risk_flags:
                self.risk_flags.append(kw)

    def recall(self, query: str, embed_fn: Optional[EmbedFn], top_k: int = 5) -> List[Tuple[str, str]]:
        """记忆召回：按与 query 的相似度返回 top_k 事实 (key, value)。"""
        scored = []
        for f in self.facts.values():
            if f.embedding is not None and embed_fn is not None:
                try:
                    q = embed_fn([query])[0]
                    v = np.asarray(f.embedding, dtype=np.float32)
                    cos = float(np.dot(q, v) / ((np.linalg.norm(q) * np.linalg.norm(v)) + 1e-9))
                except Exception:  # noqa: BLE001
                    cos = 0.0
            else:
                # 无向量时退化为简单包含匹配
                cos = 1.0 if (f.key in query or any(w in query for w in f.value[:20])) else 0.0
            scored.append((cos, f.key, f.value))
        scored.sort(key=lambda x: -x[0])
        return [(k, v) for _, k, v in scored[:top_k]]

    # ---------------- 序列化 ---------------- #
    def to_card_text(self) -> str:
        """渲染为注入 prompt 的案件记忆卡片。"""
        lines = []
        if self.case_type:
            lines.append(f"案件类型: {self.case_type}")
        for f in self.facts.values():
            lines.append(f"- {f.key}: {f.value}")
        if self.demands:
            lines.append(f"用户诉求: {'；'.join(self.demands)}")
        if self.evidence:
            lines.append(f"已有证据: {'；'.join(self.evidence)}")
        lines.append(f"处理阶段: {self.stage} | 事实完整度: {self.completeness():.0%}")
        if self.risk_flags:
            lines.append(f"[风险标记] {'、'.join(self.risk_flags)} —— 回答须包含安全提示")
        for c in self.unresolved_conflicts():
            lines.append(f"[待确认冲突] {c.key}: 「{c.old_value}」vs「{c.new_value}」")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "case_type": self.case_type,
            "facts": {k: {"value": f.value, "turn": f.source_turn, "id": f.fact_id} for k, f in self.facts.items()},
            "evidence": self.evidence,
            "stage": self.stage,
            "demands": self.demands,
            "risk_flags": self.risk_flags,
            "conflicts": [
                {"key": c.key, "old": c.old_value, "new": c.new_value, "resolved": c.resolved}
                for c in self.conflicts
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MidTermMemory":
        m = cls()
        m.case_type = d.get("case_type", "")
        m.evidence = list(d.get("evidence", []))
        m.stage = d.get("stage", "咨询初期")
        m.demands = list(d.get("demands", []))
        m.risk_flags = list(d.get("risk_flags", []))
        for k, f in d.get("facts", {}).items():
            m.facts[k] = Fact(k, f.get("value", ""), int(f.get("turn", 0)))
        for c in d.get("conflicts", []):
            m.conflicts.append(
                ConflictRecord(c["key"], c.get("old", ""), c.get("new", ""), resolved=bool(c.get("resolved")))
            )
        return m


class LongTermMemory:
    """长期记忆：用户画像（跨会话持久化）。"""

    def __init__(self, user_id: str = "default") -> None:
        self.user_id = user_id
        self.profile: Dict[str, str] = {}   # 如 地域/身份/偏好

    def update(self, kv: Dict[str, str]) -> None:
        for k, v in kv.items():
            if v and str(v).strip():
                self.profile[k] = str(v).strip()

    def to_text(self) -> str:
        if not self.profile:
            return ""
        return "；".join(f"{k}={v}" for k, v in self.profile.items())

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"user_id": self.user_id, "profile": self.profile}, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "LongTermMemory":
        p = Path(path)
        if not p.exists():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        m = cls(d.get("user_id", "default"))
        m.profile = dict(d.get("profile", {}))
        return m


class HierarchicalMemory:
    """分层记忆总控（短期 + 中期 + 长期）。

    短期记忆只保留最近 window 轮原文；更早的对话依赖中期记忆的结构化沉淀，
    这正是分层设计控制上下文长度的机制（消融实验中"扁平记忆"的对照即：
    直接拼接全部历史原文，见 runner 的 ablation 开关）。
    """

    def __init__(self, user_id: str = "default", window: int = 6, embed_fn: Optional[EmbedFn] = None) -> None:
        self.user_id = user_id
        self.window = window
        self.embed_fn = embed_fn
        self.short_turns: List[Dict[str, str]] = []   # [{"role": "user"/"assistant", "content": ...}]
        self.mid = MidTermMemory()
        self.long = LongTermMemory(user_id)

    # ---------------- 短期 ---------------- #
    def append_turn(self, role: str, content: str) -> None:
        self.short_turns.append({"role": role, "content": content, "ts": time.strftime("%H:%M:%S")})
        # 风险词进入中期记忆的风险标记
        if role == "user":
            self.mid.scan_risk(content)

    def short_window_text(self) -> str:
        turns = self.short_turns[-self.window * 2 :]  # user+assistant 各 window 条
        return "\n".join(f"{'用户' if t['role'] == 'user' else '助手'}: {t['content']}" for t in turns)

    # ---------------- 组装 prompt 记忆区 ---------------- #
    def render_prompt_sections(self, query: str, recall_top_k: int = 5) -> str:
        sections = []
        lt = self.long.to_text()
        if lt:
            sections.append(f"【用户画像（长期记忆）】{lt}")
        recalled = self.mid.recall(query, self.embed_fn, top_k=recall_top_k)
        card = self.mid.to_card_text()
        if card:
            shown = set(k for k, _ in recalled)
            sections.append("【案件记忆（中期记忆）】\n" + card)
            if recalled and shown:
                sections.append("（与当前问题最相关的记忆: " + "；".join(f"{k}" for k, _ in recalled) + "）")
        hist = self.short_window_text()
        if hist:
            sections.append("【近期对话（短期记忆）】\n" + hist)
        return "\n\n".join(sections)

    # ---------------- 持久化 ---------------- #
    def save(self, out_dir: str) -> None:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.user_id}_mid.json").write_text(
            json.dumps(self.mid.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        self.long.save(str(d / f"{self.user_id}_long.json"))

    def load(self, out_dir: str) -> None:
        d = Path(out_dir)
        mid_p = d / f"{self.user_id}_mid.json"
        if mid_p.exists():
            self.mid = MidTermMemory.from_dict(json.loads(mid_p.read_text(encoding="utf-8")))
        long_p = d / f"{self.user_id}_long.json"
        if long_p.exists():
            self.long = LongTermMemory.load(str(long_p))
