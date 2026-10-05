"""Runtime prompts, including the current-turn scope correction."""

def format_agent_action(tool_name, tool_args):
    """Format the agent action section for the verifier prompt."""
    return f"The agent wants to execute: {tool_name}({tool_args})"


def build_checklist(requirements: list) -> str:
    """Build checklist section from policy requirements list."""
    lines = []
    for req in requirements:
        name = req.get('name', '?')
        verification = req.get('verification', '')
        hint = f" (check: {verification})" if verification else ""
        lines.append(f"- {name}: [MET/NOT MET] — [evidence]{hint}")
    return "\n".join(lines)


RAW_POLICY_PROMPT = """You are a policy compliance verifier for an airline customer service agent.

## Agent Action
{agent_action}

## Policy
{policy}

## General Rules
{general_rules}

## Conversation History
{message_history}

## Task
Check whether this action complies with the policy. You MUST evaluate EVERY requirement listed below.

Evaluate the one-tool-at-a-time rule only for the assistant turn containing the proposed call; violations in earlier turns are not grounds for blocking the current call.

Only trust values confirmed by tool call results (get_user_details, get_reservation_details, search results, etc.), not user-claimed values. If a required action was never performed in the conversation, treat it as NOT MET.

## Output Format

You MUST check each requirement and output in this exact format:

CHECKLIST:
{checklist}

After completing ALL checks above:
- If ALL are MET → output: VERDICT: PASS
- If ANY is NOT MET → output:
  VERDICT: BLOCK
  AGENT_MESSAGE:
  [1-3 sentences in plain language. Tell the agent what is wrong and what to do next. Use actual values from the conversation.]"""


HYBRID_ADVISORY_PROMPT = """You are a policy compliance verifier for an airline customer service agent.

## Agent Action
{agent_action}

## Full Policy Document (authoritative source of truth)
{raw_policy}

## Per-Tool Policy (additional structured detail)
{policy}

## General Rules
{general_rules}

## Conversation History
{message_history}

## Task
Decide whether this action complies with the policy as a whole. The checklist
below lists pre-identified focal points for the target tool — they are a
guide, not a gate. A single NOT MET does not automatically force BLOCK if the
full policy does not require it in this context. Conversely, the policy may
require something the checklist does not list — apply it.

Evaluate the one-tool-at-a-time rule only for the assistant turn containing
the proposed call; violations in earlier turns are not grounds for blocking
the current call.

Only trust values confirmed by tool call results, not user-claimed values. If
a required action was never performed in the conversation, treat that
requirement as NOT MET.

## Output Format

You MUST evaluate each checklist item, then deliver a holistic verdict:

CHECKLIST (advisory — note MET/NOT MET/N/A and your reasoning):
{checklist}

REASONING:
[Brief synthesis: which items genuinely matter for this action, which are
N/A in context, and whether the action complies with the policy as a whole.]

Then output one of:
- VERDICT: PASS
- VERDICT: BLOCK
  AGENT_MESSAGE:
  [1-3 sentences in plain language. Tell the agent what is wrong and what to do next. Use actual values from the conversation.]"""
