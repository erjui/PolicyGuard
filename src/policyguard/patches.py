"""Monkey-patches for tau2 orchestrator to support PolicyGuard verification."""

import json
from typing import Optional

from tau2.orchestrator.orchestrator import Orchestrator, Role
from tau2.data_model.message import MultiToolMessage, ToolMessage


def patch_orchestrator(verifier, debug_log_path: Optional[str] = None):
    """Patch Orchestrator.step to verify mutating tool calls before env execution.

    In AGENT → ENV path: check verifier before env.get_response().
    If blocked: pop tool_call from trajectory, route error back to agent.
    Uses self.trajectory (per-instance) — safe for concurrent execution.
    """
    original_step = Orchestrator.step

    def _debug_log(entry: dict):
        if not debug_log_path:
            return
        with open(debug_log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def patched_step(self):
        # Only intercept AGENT → ENV
        if not (self.from_role == Role.AGENT and self.to_role == Role.ENV):
            original_step(self)
            return

        if not self.message.is_tool_call():
            original_step(self)
            return

        # Check each tool call against verifier (trajectory is per-instance, concurrency safe)
        messages = list(self.trajectory)
        blocked_tc = None
        verdict = None

        for tc in self.message.tool_calls:
            if tc.requestor == "assistant" and tc.name in verifier.MUTATING_TOOLS:
                verdict = verifier.check_tool_call(tc.name, tc.arguments, messages,
                                                   task_id=str(self.task.id))

                _debug_log({
                    "event": "verifier_check",
                    "task_id": str(self.task.id),
                    "tool": tc.name,
                    "tool_args": tc.arguments,
                    "passed": verdict.passed,
                    "agent_message": verdict.agent_message or "",
                    "reasoning": verdict.reasoning or "",
                })

                if verdict.is_warning():
                    blocked_tc = tc
                    break

        if blocked_tc is None:
            # All passed — run original step (env executes)
            original_step(self)
            return

        # BLOCKED — pop tool_call from trajectory, route error back to agent
        self.trajectory.pop()  # remove the AssistantMessage with tool_calls

        # Create ToolMessage(error) for each tool_call_id (OpenAI API requires responses for all)
        tool_msgs = []
        for tc in self.message.tool_calls:
            if tc == blocked_tc:
                tool_msgs.append(ToolMessage(
                    id=tc.id,
                    role="tool",
                    content=verdict.format_warning(),
                    requestor=tc.requestor,
                    error=True,
                ))
            else:
                tool_msgs.append(ToolMessage(
                    id=tc.id,
                    role="tool",
                    content="Not executed: another tool call was blocked by policy verification.",
                    requestor=tc.requestor,
                    error=True,
                ))

        # Route back to agent (NOT added to trajectory — just routing)
        if len(tool_msgs) > 1:
            self.message = MultiToolMessage(role="tool", tool_messages=tool_msgs)
        else:
            self.message = tool_msgs[0]

        self.to_role = Role.AGENT
        self.from_role = Role.ENV
        self.step_count += 1
        self.environment.sync_tools()

    Orchestrator.step = patched_step
