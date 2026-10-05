"""Dialogue-grounded verification of policy-sensitive tool calls."""


from dataclasses import dataclass
from typing import List, Dict
from tau2.data_model.message import Message, SystemMessage
from tau2.utils.llm_utils import generate
from policyguard.prompts import RAW_POLICY_PROMPT, HYBRID_ADVISORY_PROMPT, format_agent_action, build_checklist
from policyguard.policy_loader import get_policy_for_tool, format_policy_for_prompt


MAX_BLOCKS_PER_TOOL = 5


MUTATING_TOOLS_BY_DOMAIN = {
    "airline": {
        "cancel_reservation", "book_reservation",
        "update_reservation_flights", "update_reservation_passengers",
        "update_reservation_baggages", "send_certificate",
    },
    # Populated from generator's --classify-only on gpt-5.4 (2026-05-08).
    # `transfer_to_human_agents` excluded per airline convention: it is an
    # escalation, not a DB mutation, and gating it would over-block.
    "retail": {
        "cancel_pending_order", "exchange_delivered_order_items",
        "modify_pending_order_address", "modify_pending_order_items",
        "modify_pending_order_payment", "modify_user_address",
        "return_delivered_order_items",
    },
    "telecom": {
        "disable_roaming", "enable_roaming", "refuel_data",
        "resume_line", "send_payment_request", "suspend_line",
    },
}


def get_mutating_tools(domain: str) -> set:
    if domain not in MUTATING_TOOLS_BY_DOMAIN:
        raise ValueError(f"Unknown domain: {domain!r}. "
                         f"Known: {sorted(MUTATING_TOOLS_BY_DOMAIN)}")
    return MUTATING_TOOLS_BY_DOMAIN[domain]


@dataclass
class Verdict:
    passed: bool
    reasoning: str = ""
    agent_message: str = ""

    def is_warning(self) -> bool:
        return not self.passed

    def format_warning(self) -> str:
        """Returns agent-facing message only."""
        return self.agent_message if not self.passed else ""


def _parse_verdict(response_text: str) -> Verdict:
    """Parse verifier LLM output into a Verdict.

    Expected format:
        REASONING: [analysis]
        VERDICT: PASS
    or:
        REASONING: [analysis]
        VERDICT: BLOCK
        AGENT_MESSAGE: [error for agent]
    """
    text = response_text.strip()

    # Check for VERDICT: PASS / BLOCK
    if "VERDICT: PASS" in text:
        reasoning = text.split("VERDICT:")[0].replace("REASONING:", "").strip()
        return Verdict(passed=True, reasoning=reasoning)

    if "VERDICT: BLOCK" in text:
        parts = text.split("VERDICT: BLOCK", 1)
        reasoning = parts[0].replace("REASONING:", "").strip()
        agent_message = ""
        remainder = parts[1] if len(parts) > 1 else ""
        if "AGENT_MESSAGE:" in remainder:
            agent_message = remainder.split("AGENT_MESSAGE:", 1)[1].strip()
        else:
            agent_message = remainder.strip()
        return Verdict(passed=False, reasoning=reasoning, agent_message=agent_message)

    # Fallback: no VERDICT keyword — legacy format or LLM didn't follow format
    if text.upper().startswith("PASS"):
        return Verdict(passed=True)

    # Treat as block with full text as agent message
    return Verdict(passed=False, agent_message=text)


def _format_history_shared(messages: List[Message], full: bool = False) -> str:
    """Format conversation history for verifier prompt (shared helper).

    Verifier-view alignment: the verifier should reason from the same
    information set the agent had access to. In tau2's dual-control telecom
    domain, the orchestrator's trajectory contains user-side tool calls and
    tool results that are NEVER routed to the agent (agent only sees the
    user's natural-language messages). We filter those out here so the
    verifier evaluates the agent's decisions on the agent's evidence —
    avoiding both unfair verdicts (verifier sees ground truth the agent
    couldn't have seen) and benchmark-specific overfitting (a verifier that
    relies on user-side tool results does not generalize to deployment).

    For airline / retail (no user_tools.py), this filter is a no-op.

    """
    lines = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        role = getattr(msg, "role", "?")
        tool_calls = getattr(msg, "tool_calls", None)
        content = getattr(msg, "content", None)
        requestor = getattr(msg, "requestor", None)

        # Drop user-side tool activity — the agent never receives these.
        if tool_calls and role == "user":
            continue
        if role == "tool" and requestor == "user":
            continue

        if tool_calls:
            for tc in tool_calls:
                args_str = str(tc.arguments)
                if not full and len(args_str) > 700:
                    args_str = args_str[:700] + "..."
                lines.append(f"AGENT->TOOL: {tc.name}({args_str})")
        elif content:
            if full:
                text = content
            else:
                text = content if len(content) <= 1500 else content[:1500] + "..."
            if role == "tool":
                error = getattr(msg, "error", False)
                prefix = "TOOL_ERROR" if error else "TOOL_RESULT"
                lines.append(f"{prefix}: {text}")
            else:
                lines.append(f"{role.upper()}: {text}")
    if full:
        return "\n".join(lines)
    return "\n".join(lines[-50:])


class PolicyGuardHybridVerifier:
    """Verify the raw policy using an advisory per-tool checklist."""

    def __init__(self, policies: Dict[str, dict], raw_policy: str,
                 model: str = "gpt-4o-mini",
                 full_history: bool = True, domain: str = "airline"):
        self.policies = policies
        self.general_rules = policies.get("general_rules", {})
        self.raw_policy = raw_policy
        self.model = model
        self.full_history = full_history
        self.domain = domain
        self.MUTATING_TOOLS = get_mutating_tools(domain)
        self._block_counts: Dict[str, int] = {}

    def check_tool_call(self, tool_name: str, tool_args: dict,
                        messages: List[Message],
                        task_id: str = "") -> Verdict:
        if tool_name not in self.MUTATING_TOOLS:
            return Verdict(passed=True)

        intent_policy = get_policy_for_tool(self.policies, tool_name)
        if not intent_policy:
            return Verdict(passed=True)

        block_key = f"{task_id}:{tool_name}"
        if self._block_counts.get(block_key, 0) >= MAX_BLOCKS_PER_TOOL:
            return Verdict(
                passed=False,
                agent_message=f"Tool {tool_name} has been blocked {self._block_counts[block_key]} times. "
                              f"Unable to complete this action. Please transfer the user to a human agent.",
            )

        history = self._format_history(messages, full=self.full_history)
        args_str = str(tool_args)
        if not self.full_history and len(args_str) > 700:
            args_str = args_str[:700] + "..."

        if "requirements" not in intent_policy:
            checklist = "(no checklist available — check all policy requirements)"
        else:
            checklist = build_checklist(intent_policy['requirements'])

        prompt = HYBRID_ADVISORY_PROMPT.format(
            agent_action=format_agent_action(tool_name, args_str),
            raw_policy=self.raw_policy,
            policy=format_policy_for_prompt(intent_policy),
            general_rules=format_policy_for_prompt(self.general_rules),
            message_history=history,
            checklist=checklist,
        )

        response = generate(
            model=self.model,
            messages=[SystemMessage(role="system", content=prompt)],
            temperature=0.0,
        )

        verdict = _parse_verdict(response.content or "PASS")

        if verdict.is_warning():
            self._block_counts[block_key] = self._block_counts.get(block_key, 0) + 1
        else:
            self._block_counts.pop(block_key, None)

        return verdict

    def _format_history(self, messages: List[Message],
                        full: bool = False) -> str:
        return _format_history_shared(messages, full)


class PolicyGuardRawVerifier:
    """Verify against the raw domain policy and the full agent-visible dialogue."""

    def __init__(self, raw_policy: str, model: str = "gpt-4o-mini",
                 full_history: bool = True, domain: str = "airline"):
        self.raw_policy = raw_policy
        self.model = model
        self.full_history = full_history
        self.domain = domain
        self.MUTATING_TOOLS = get_mutating_tools(domain)
        self._block_counts: Dict[str, int] = {}

    def check_tool_call(self, tool_name: str, tool_args: dict,
                        messages: List[Message],
                        task_id: str = "") -> Verdict:
        if tool_name not in self.MUTATING_TOOLS:
            return Verdict(passed=True)

        block_key = f"{task_id}:{tool_name}"
        if self._block_counts.get(block_key, 0) >= MAX_BLOCKS_PER_TOOL:
            return Verdict(
                passed=False,
                agent_message=f"Tool {tool_name} has been blocked {self._block_counts[block_key]} times. "
                              f"Unable to complete this action. Please transfer the user to a human agent.",
            )

        history = _format_history_shared(
            messages, full=self.full_history,
        )
        args_str = str(tool_args)
        if not self.full_history and len(args_str) > 700:
            args_str = args_str[:700] + "..."

        prompt = RAW_POLICY_PROMPT.format(
            agent_action=format_agent_action(tool_name, args_str),
            policy=self.raw_policy,
            general_rules="N/A",
            message_history=history,
            checklist="(no checklist — check all policy requirements from the policy text above)",
        )

        response = generate(
            model=self.model,
            messages=[SystemMessage(role="system", content=prompt)],
            temperature=0.0,
        )

        verdict = _parse_verdict(response.content or "PASS")

        if verdict.is_warning():
            self._block_counts[block_key] = self._block_counts.get(block_key, 0) + 1
        else:
            self._block_counts.pop(block_key, None)

        return verdict
