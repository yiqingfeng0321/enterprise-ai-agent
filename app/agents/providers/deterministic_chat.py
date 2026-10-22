from __future__ import annotations

import json
import re

from app.models.domain import ChatHistoryMessage, ChatResult
from app.tools.retention_tools import RetentionToolRegistry


class DeterministicRetentionChatProvider:
    """CI-safe conversational fallback for common retention questions."""

    name = "deterministic_chat"
    model_id = "rules-chat-v1"
    generation_mode = "deterministic"
    service_available = True

    def __init__(self, tools: RetentionToolRegistry):
        self.tools = tools

    @staticmethod
    def _limit(message: str) -> int:
        patterns = (
            r"\btop\s*(\d{1,3})\b",
            r"前\s*(\d{1,3})\s*(?:个|名)?",
            r"最高(?:的)?\s*(\d{1,3})\s*(?:个|名)?",
        )
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return min(max(int(match.group(1)), 1), 100)
        return 10

    @staticmethod
    def _customer_id(message: str) -> str | None:
        explicit = re.search(r"\bCUST[- ]?(\d{1,4})\b", message, re.IGNORECASE)
        if explicit:
            return f"CUST-{int(explicit.group(1)):04d}"
        shorthand = re.search(
            r"(?:客户|customer)\s*(?:id)?\s*[-:#：]?\s*0*(\d{1,4})",
            message,
            re.IGNORECASE,
        )
        if shorthand:
            return f"CUST-{int(shorthand.group(1)):04d}"
        leading = re.search(r"0*(\d{1,4})\s*(?:号)?客户", message, re.IGNORECASE)
        if leading:
            return f"CUST-{int(leading.group(1)):04d}"
        return None

    @staticmethod
    def _markdown_table(rows: list[dict]) -> str:
        lines = [
            "| Rank | Customer | Risk | Band | Segment | Renewal | Usage change |",
            "|---:|---|---:|---|---|---:|---:|",
        ]
        for index, row in enumerate(rows, start=1):
            lines.append(
                "| {rank} | {customer} | {risk:.1%} | {band} | {segment} | "
                "{renewal} days | {usage:+.1f}% |".format(
                    rank=index,
                    customer=row["customer_id"],
                    risk=float(row["churn_probability"]),
                    band=row["risk_band"],
                    segment=row["segment"],
                    renewal=row["days_to_renewal"],
                    usage=float(row["usage_change_pct"]),
                )
            )
        return "\n".join(lines)

    async def chat(
        self,
        message: str,
        history: list[ChatHistoryMessage] | None = None,
    ) -> ChatResult:
        del history
        lowered = message.lower()
        ranking_tokens = ("top", "最高", "风险客户", "高风险", "排名", "排行")
        if any(token in lowered for token in ranking_tokens) and not self._customer_id(message):
            limit = self._limit(message)
            rows = json.loads(self.tools.list_high_risk_customers(limit=limit))
            answer = (
                f"下面是按当前模型 churn probability 排序的 Top {len(rows)} 客户。"
                "这是风险优先级，不是已经发生流失的事实。\n\n"
                f"{self._markdown_table(rows)}"
            )
            return ChatResult(
                answer=answer,
                provider=self.name,
                model=self.model_id,
                generation_mode=self.generation_mode,
                tools_used=["list_high_risk_customers"],
            )

        customer_id = self._customer_id(message)
        if customer_id:
            profile = json.loads(self.tools.get_customer_profile(customer_id))
            prediction = json.loads(self.tools.predict_churn(customer_id))
            memories = json.loads(self.tools.get_customer_memories(customer_id))
            tools_used = ["get_customer_profile", "predict_churn", "get_customer_memories"]

            if any(token in lowered for token in ("折扣", "discount", "打折")):
                knowledge = json.loads(
                    self.tools.search_retention_playbooks(
                        "discount renewal adoption recovery value realization"
                    )
                )
                tools_used.append("search_retention_playbooks")
                reasons: list[str] = []
                if float(profile["active_seat_ratio_30d"]) < 0.40:
                    reasons.append(
                        f"活跃席位率只有 {float(profile['active_seat_ratio_30d']):.1%}，"
                        "风险更像 adoption 问题，降价不会直接修复使用障碍。"
                    )
                if float(profile["usage_change_pct"]) <= -15:
                    reasons.append(
                        f"近 30 天使用变化为 {float(profile['usage_change_pct']):+.1f}%，"
                        "应先定位工作流和部署阻塞。"
                    )
                if int(profile["support_tickets_90d"]) >= 4:
                    reasons.append(
                        f"90 天内有 {int(profile['support_tickets_90d'])} 个支持工单，"
                        "优先解决服务或技术根因更合理。"
                    )
                if int(profile["days_to_renewal"]) > 120:
                    reasons.append(
                        f"距离续约还有 {int(profile['days_to_renewal'])} 天，"
                        "现在给折扣可能过早。"
                    )
                prior_discount = [
                    item["content"]
                    for item in memories
                    if "discount" in item["content"].lower() or "折扣" in item["content"]
                ]
                if prior_discount:
                    reasons.append("历史 Memory 已记录折扣相关干预，因此不应在没有效果证据时重复。")
                if not reasons:
                    reasons.append(
                        "现有数据没有证明折扣是最佳干预；应先验证风险根因和客户价值缺口。"
                    )
                evidence = "\n".join(f"- {reason}" for reason in reasons)
                source_note = (
                    f"检索到 {len(knowledge)} 条 approved playbook 片段。"
                    if knowledge
                    else "当前检索未返回相关 playbook，因此结论只基于客户和模型事实。"
                )
                answer = (
                    f"并不是说 **{customer_id} 永远不能打折**，而是当前证据不支持把折扣作为第一选择。\n\n"
                    f"模型风险：**{float(prediction['churn_probability']):.1%} "
                    f"({prediction['risk_band']})**。\n\n{evidence}\n\n"
                    f"{source_note}\n\n"
                    "更合理的顺序是先处理 adoption/support/value-realization 根因，"
                    "再由 account owner 审批是否需要商业让步。"
                )
                return ChatResult(
                    answer=answer,
                    provider=self.name,
                    model=self.model_id,
                    generation_mode=self.generation_mode,
                    tools_used=tools_used,
                )

            answer = (
                f"**{customer_id}** 当前模型风险为 "
                f"**{float(prediction['churn_probability']):.1%} ({prediction['risk_band']})**。 "
                f"客户属于 {profile['segment']}，距离续约 {profile['days_to_renewal']} 天，"
                f"使用变化 {float(profile['usage_change_pct']):+.1f}%，"
                f"活跃席位率 {float(profile['active_seat_ratio_30d']):.1%}。 "
                f"系统中有 {len(memories)} 条相关 Memory。"
            )
            return ChatResult(
                answer=answer,
                provider=self.name,
                model=self.model_id,
                generation_mode=self.generation_mode,
                tools_used=tools_used,
            )

        return ChatResult(
            answer=(
                "我是客户留存领域 Copilot。你可以问我：\n\n"
                "- 风险最高的 10 个客户是谁？\n"
                "- 为什么 02 客户不应该先使用折扣？\n"
                "- CUST-0042 的风险和历史干预是什么？\n"
                "- 哪类 retention playbook 适合 adoption 下滑？\n\n"
                "客户或组合问题会通过受控 Tools 查询；我不会编造客户事实。"
            ),
            provider=self.name,
            model=self.model_id,
            generation_mode=self.generation_mode,
            tools_used=[],
        )
