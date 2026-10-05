"""Prompt templates for LLM-based structured policy generation.

Four-step pipeline:
1. Classify tools as mutating vs read-only (1 LLM call)
2. Generate policy YAML per mutating tool (N LLM calls)
3. Generate general_rules.yaml (1 LLM call)
4. Review all files for completeness against wiki (1 LLM call)

Total: N+3 calls, where N = number of mutating tools identified.
"""

SYSTEM_PROMPT = """\
You are a policy engineer for PolicyGuard, a pre-execution verification system \
for AI agent tool calls. Your task is to convert a raw, unstructured policy \
document into structured YAML policy files.

PolicyGuard uses these YAML files to verify that an AI agent has completed all \
required steps before executing a mutating tool call. Each tool has its own \
policy file specifying requirements that must be met before the tool is called.

Be thorough and precise:
- Every rule in the source document must be captured in the output.
- Do NOT invent rules that are not in the source document.
- Do NOT omit rules because they seem obvious.
- Capture nuanced conditions exactly (tier-dependent logic, conditional \
eligibility, branching rules).\
"""

# --- Step 1: Classify tools ---

CLASSIFY_PROMPT_TEMPLATE = """\
## Task

Given the policy document and tool signatures below, classify each tool as \
either **mutating** (changes state — e.g., creating, updating, deleting, \
issuing, transferring) or **read-only** (informational — e.g., searching, \
looking up, calculating).

## Raw Policy Document

```
{wiki_text}
```

## Available Tool Signatures

```
{tool_signatures}
```

## Output Format

Respond with a JSON object:

```json
{{
  "mutating": ["tool_name_1", "tool_name_2", ...],
  "read_only": ["tool_name_3", "tool_name_4", ...]
}}
```

Classify ALL tools. Every tool must appear in exactly one list.\
"""

# --- Step 2: Per-tool policy generation ---

TOOL_POLICY_PROMPT_TEMPLATE = """\
## Task

Generate a structured YAML policy file for the tool `{tool_name}`.

Read the policy document carefully and extract ALL requirements \
that apply to this specific tool. The policy file will be used by a verifier \
to check that an AI agent has completed all required steps before calling \
this tool.

## Raw Policy Document

```
{wiki_text}
```

## All Available Tools

```
{tool_signatures}
```

## Target Tool: `{tool_name}`

## YAML Schema

```yaml
tool: {tool_name}
description: <one-line description of what this tool does>
requirements:
  - name: <step_name_in_snake_case>
    # Describe the step using ONE primary field:
    method: <how the agent should perform this step>     # conversational actions
    tool: <tool_name>                                     # tool-based steps
    logic: |                                              # complex decision logic
      <multi-line conditional logic>
    details: <description of what is needed>              # informational steps
    rule: <the rule to follow>                            # confirmation/rule steps
    # Optional fields (include only when relevant):
    purpose: <why this step exists>
    condition: <when this requirement applies — for conditional steps>
    constraints:                                          # sub-constraints for this step
      - <constraint>
    valid_values: [<list of acceptable values>]
    on_ineligible: <action when eligibility check fails>
    on_violation: <action when rule is violated>
    # REQUIRED for every requirement:
    verification: <how a verifier can confirm this step was completed \
from conversation history>
```

## Guidelines

- Each true pre-call requirement must be an actionable checklist item that a \
verifier can evaluate as MET/NOT MET from conversation evidence before this \
tool executes.
- Do not strengthen the source policy: never turn a post-action instruction \
into a pre-call condition or require a particular evidence-gathering tool when \
equivalent tool-grounded evidence establishes the same policy fact.
- Requirements fall into two categories:
  1. **Data verification** (checking system state): Specify a `tool:` field \
only when the source policy mandates that exact lookup. Otherwise describe the \
required fact and accept any read-only tool result that establishes it. \
User-claimed values alone are never sufficient for data checks.
  2. **Procedural steps** (conversation actions): Use `method:` field. \
Verification checks for evidence in conversation history (user confirmations, \
agent disclosures, information collected).
- Every requirement MUST have a `verification` field.
- Use `condition:` when a requirement only applies in certain scenarios \
(requirements specific to one path should have a `condition:` field stating \
when they apply and when they can be skipped).
- Use `logic:` with multi-line `|` for complex eligibility checks with \
branching conditions.

## Instructions

Think step by step:
1. Which section(s) of the policy document apply to `{tool_name}`?
2. What requirements must be met before calling this tool?
3. Are there distinct paths or scenarios? If so, use `condition:` fields.
4. What verification evidence should a verifier look for per step?

Reason through your analysis first, then output the final YAML inside a \
```yaml block.\
"""

# --- Step 3: General rules ---

GENERAL_RULES_PROMPT_TEMPLATE = """\
## Task

Generate a `general_rules.yaml` file that captures cross-cutting rules \
and reference information from the policy document.

General rules complement per-tool policies. They include:
- **Behavioral rules** that govern the agent's general conduct
- **Agent capabilities** — what each tool can and cannot do
- **Domain reference data** — entity schemas, enumerated types, lookup \
tables, status definitions, pricing rules, and other structured facts \
from the policy document.

Some information may naturally appear in both general rules and per-tool \
policies. That is fine — general rules serve as a reference and overview, \
while per-tool policies provide actionable requirements.

## Raw Policy Document

```
{wiki_text}
```

## Available Tool Signatures

```
{tool_signatures}
```

## Mutating Tools Identified

The following tools have been classified as mutating (state-changing):
{mutating_tools_list}

## Recommended Structure

```yaml
behavioral_rules:
- name: <rule_name_in_snake_case>
  rule: <the rule>
  verification: <how to verify>

capabilities:
  can_do:
  - tool: <tool_name>
    what: <brief description>
  cannot_do:
  - <thing the system cannot do>

domain_facts:
  # Extract ALL domain-specific reference data from the policy document:
  # entity field schemas, enumerated types, lookup tables, time/timezone
  # info, pricing rules, tier/class hierarchies, status definitions,
  # and any structured data referenced by tool policies.
  <fact_category>: <value or structured data>
```

## Instructions

Think step by step:
1. What general behavioral constraints does the policy document state?
2. What are the agent's capabilities and limitations?
3. What domain reference data is mentioned?

Reason through your analysis first, then output the final YAML inside a \
```yaml block.\
"""

# --- Step 4: Review for completeness ---

REVIEW_PROMPT_TEMPLATE = """\
## Task

You are given:
1. The **original policy document** (the source of truth)
2. **Generated policy files** (general_rules.yaml + per-tool policies)

Review the generated policies for **completeness, correctness, and conciseness** \
against the original policy document:
- Is every rule from the policy document captured somewhere in the generated files?
- Are there any missing policies or requirements?
- Are rules placed appropriately?
- Should any rules be rearranged to better locations?
- **Ensure each true pre-call requirement is an actionable checklist item** — \
checkable as MET/NOT MET from conversation evidence. Data-verification \
requirements must state the required fact; name a particular read-only tool \
only when the source policy mandates it.
- **Consolidate over-split requirements.** If multiple requirements in a tool \
policy describe the same concept, merge them into a single requirement with \
sub-constraints.

## Original Policy Document (source of truth)

```
{wiki_text}
```

## Generated Files

### general_rules.yaml
```yaml
{general_rules_content}
```

{tool_policies_section}

## Instructions

Reason through carefully:
1. Go through each section of the original policy document line by line.
2. For each rule, verify it is captured in at least one generated file.
3. Note any missing rules and which file they should go in.
4. For each tool policy, verify every requirement is a concrete, verifiable \
checklist item. Data-verification requirements must specify a `tool:` field.
5. For each tool policy, identify over-split requirements (multiple steps \
for one concept) and consolidate them.
6. Check that the YAML schema is consistent (use `name:` not `id:` for \
identifiers, every requirement has `verification:`).

After your analysis, output all corrected files using this format:

--- FILE: general_rules.yaml ---
<corrected YAML content>

--- FILE: tool_name.yaml ---
<corrected YAML content>

Output ALL files (even unchanged ones). Only output the file markers and \
YAML content — no additional explanation after the files.\
"""


def build_classify_prompt(wiki_text: str, tool_signatures: str) -> str:
    """Build the tool classification prompt."""
    return CLASSIFY_PROMPT_TEMPLATE.format(
        wiki_text=wiki_text,
        tool_signatures=tool_signatures,
    )


def build_tool_policy_prompt(
    tool_name: str, wiki_text: str, tool_signatures: str
) -> str:
    """Build the per-tool policy generation prompt."""
    return TOOL_POLICY_PROMPT_TEMPLATE.format(
        tool_name=tool_name,
        wiki_text=wiki_text,
        tool_signatures=tool_signatures,
    )


def build_general_rules_prompt(
    wiki_text: str, tool_signatures: str, mutating_tools: list[str]
) -> str:
    """Build the general rules generation prompt."""
    mutating_tools_list = "\n".join(f"- `{t}`" for t in mutating_tools)
    return GENERAL_RULES_PROMPT_TEMPLATE.format(
        wiki_text=wiki_text,
        tool_signatures=tool_signatures,
        mutating_tools_list=mutating_tools_list,
    )


def build_review_prompt(
    wiki_text: str,
    general_rules_content: str,
    tool_policies: dict[str, str],
) -> str:
    """Build the review/completeness-check prompt."""
    tool_sections = []
    for tool_name, content in sorted(tool_policies.items()):
        tool_sections.append(f"### {tool_name}.yaml\n```yaml\n{content}\n```")
    tool_policies_section = "\n\n".join(tool_sections)

    return REVIEW_PROMPT_TEMPLATE.format(
        wiki_text=wiki_text,
        general_rules_content=general_rules_content,
        tool_policies_section=tool_policies_section,
    )
