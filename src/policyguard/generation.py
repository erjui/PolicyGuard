"""Generate per-tool checklists and general rules, then review against the source policy."""

import argparse
import inspect
import os
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
import yaml

from policyguard.generation_prompts import (
    SYSTEM_PROMPT,
    build_classify_prompt,
    build_tool_policy_prompt,
    build_general_rules_prompt,
    build_review_prompt,
)
from policyguard.policy_text import load_domain_policy_text


def _import_tools_class(domain: str):
    if domain == "airline":
        from tau2.domains.airline.tools import AirlineTools
        return AirlineTools
    if domain == "retail":
        from tau2.domains.retail.tools import RetailTools
        return RetailTools
    if domain == "telecom":
        from tau2.domains.telecom.tools import TelecomTools
        return TelecomTools
    raise ValueError(f"Unknown domain: {domain!r}")


def extract_tool_signatures(domain: str) -> str:
    """Extract tool signatures from tau2 <Domain>Tools."""
    Tools = _import_tools_class(domain)

    funcs = [
        member
        for _, member in inspect.getmembers(Tools, predicate=inspect.isfunction)
        if getattr(member, "__tool__", None)
    ]

    lines = []
    for fn in sorted(funcs, key=lambda f: f.__name__):
        doc = inspect.getdoc(fn) or ""
        lines.append(f"### {fn.__name__}")
        lines.append(doc)
        lines.append("")

    return "\n".join(lines)


def call_llm(client: OpenAI, model: str, temperature: float,
             system: str, user: str) -> tuple[str, dict]:
    """Make a single LLM call. Returns (response_text, usage_dict)."""
    kwargs = dict(
        model=model, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    try:
        response = client.chat.completions.create(**kwargs)
    except BadRequestError as exc:
        if "temperature" not in str(exc):
            raise
        kwargs.pop("temperature")
        response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return text, usage


def parse_classification(text: str) -> dict:
    """Extract JSON classification from LLM response."""
    # Find JSON block (possibly wrapped in ```json fences)
    json_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    # Try parsing entire response as JSON
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))
    raise ValueError(f"Could not parse classification JSON from response:\n{text}")


def strip_yaml_fences(text: str) -> str:
    """Strip ```yaml fences if LLM wrapped content."""
    text = text.strip()
    text = re.sub(r"^```ya?ml\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text


def extract_yaml_block(text: str) -> str:
    """Extract YAML content from a response that may include reasoning.

    The LLM is asked to reason first, then output YAML in a ```yaml block.
    This extracts the YAML block from within the reasoning.
    Falls back to strip_yaml_fences if no fenced block found.
    """
    match = re.search(r"```ya?ml\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: treat entire response as YAML
    return strip_yaml_fences(text)


def parse_review_output(text: str) -> dict[str, str]:
    """Parse review response into {filename: yaml_content} dict.

    Expected format (after optional reasoning):
    --- FILE: general_rules.yaml ---
    <yaml content>

    --- FILE: book_reservation.yaml ---
    <yaml content>
    """
    files = {}
    # Split on file markers
    parts = re.split(r"---\s*FILE:\s*(\S+\.yaml)\s*---", text)
    # parts[0] is preamble (reasoning), then alternating name/content
    for i in range(1, len(parts), 2):
        filename = parts[i].strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*\.yaml", filename):
            raise ValueError(f"Invalid reviewed filename: {filename!r}")
        if filename in files:
            raise ValueError(f"Duplicate reviewed filename: {filename}")
        content = extract_yaml_block(parts[i + 1]) if i + 1 < len(parts) else ""
        files[filename] = content.strip()
    return files


def validate_policy(content: str, tool_name: str | None = None) -> None:
    """Reject malformed model output before it becomes a runtime policy file."""
    policy = yaml.safe_load(content)
    if not isinstance(policy, dict) or not policy:
        raise ValueError("Policy YAML must be a nonempty mapping")
    if tool_name is None:
        return
    if policy.get("tool") != tool_name:
        raise ValueError(f"Policy must name tool {tool_name!r}")
    requirements = policy.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError(f"{tool_name}: requirements must be a nonempty list")
    for req in requirements:
        if not isinstance(req, dict) or any(
            not isinstance(req.get(key), str) or not req[key].strip()
            for key in ("name", "verification")
        ):
            raise ValueError(f"{tool_name}: every requirement needs name and verification")


def validate_classification(classification: dict, domain: str) -> None:
    from policyguard.verifier import get_mutating_tools

    tools = {
        name for name, member in inspect.getmembers(_import_tools_class(domain), inspect.isfunction)
        if getattr(member, "__tool__", None)
    }
    if not isinstance(classification, dict):
        raise ValueError("Classification must be a JSON object")
    groups = [classification.get(key) for key in ("mutating", "read_only")]
    if any(not isinstance(group, list) for group in groups):
        raise ValueError("Classification needs mutating and read_only lists")
    names = groups[0] + groups[1]
    if any(not isinstance(name, str) or name not in tools for name in names):
        raise ValueError("Classification contains an unknown tool")
    if len(names) != len(set(names)) or set(names) != tools:
        raise ValueError("Classification must list every tool exactly once")
    missing = get_mutating_tools(domain) - set(groups[0])
    if missing:
        raise ValueError(f"Classification omits runtime-protected tools: {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured YAML policies from raw policy text using an LLM"
    )
    parser.add_argument("--domain", default="airline",
                        choices=["airline", "retail", "telecom"],
                        help="tau2 domain (default: airline)")
    parser.add_argument("--model", default="gpt-5.4",
                        help="LLM model name (default: gpt-5.4)")
    parser.add_argument("--wiki", default=None,
                        help="Override path to raw policy document. "
                             "If unset, loaded from TAU2_DATA_DIR/tau2/domains/<domain>/...")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (auto-generated if not set)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="LLM temperature (default: 0.2)")
    parser.add_argument("--classify-only", action="store_true",
                        help="Run only Step 1 (tool classification) and exit. "
                             "Useful for populating MUTATING_TOOLS_BY_DOMAIN.")
    args = parser.parse_args()

    project_root = ROOT
    load_dotenv(ROOT / ".env")
    data_dir = Path(os.environ.get("TAU2_DATA_DIR", ROOT / "external/tau2-bench/data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    os.environ["TAU2_DATA_DIR"] = str(data_dir.resolve())
    if args.wiki is not None:
        wiki_path = project_root / args.wiki
        wiki_text = wiki_path.read_text()
        wiki_label = str(wiki_path)
    else:
        wiki_text = load_domain_policy_text(args.domain, data_dir)
        wiki_label = f"<assembled from data/tau2/domains/{args.domain}>"

    print(f"Extracting tool signatures from tau2.domains.{args.domain}...")
    tool_signatures = extract_tool_signatures(args.domain)

    # Output directory
    if args.output_dir is None:
        model_tag = re.sub(r"[^A-Za-z0-9_-]", "_", args.model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / f"outputs/checklists/{args.domain}_{model_tag}_{timestamp}"
    else:
        output_dir = Path(args.output_dir)

    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Domain: {args.domain}")
    print(f"Model:  {args.model}")
    print(f"Wiki:   {wiki_label}")
    print(f"Output: {output_dir}")
    print(f"Temp:   {args.temperature}")

    client = OpenAI()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_prompts = []  # for reproducibility log

    def track_usage(usage: dict):
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    # ── Step 1: Classify tools ──
    print("\n[Step 1] Classifying tools...")
    classify_prompt = build_classify_prompt(wiki_text, tool_signatures)
    all_prompts.append(("Step 1: Classify", classify_prompt))

    classify_response, usage = call_llm(
        client, args.model, args.temperature, SYSTEM_PROMPT, classify_prompt
    )
    track_usage(usage)

    classification = parse_classification(classify_response)
    validate_classification(classification, args.domain)
    mutating_tools = classification.get("mutating", [])
    read_only_tools = classification.get("read_only", [])

    print(f"  Mutating ({len(mutating_tools)}): {mutating_tools}")
    print(f"  Read-only ({len(read_only_tools)}): {read_only_tools}")

    # Save classification
    (output_dir / "_classification.json").write_text(
        json.dumps(classification, indent=2)
    )

    if args.classify_only:
        print("\n[--classify-only] Stopping after Step 1.")
        print(f"Mutating set for {args.domain}:")
        print(f"  {sorted(mutating_tools)}")
        print(f"\nPaste this into MUTATING_TOOLS_BY_DOMAIN[{args.domain!r}] "
              f"in src/policyguard/verifier.py.")
        return

    # ── Step 2: Generate policy per mutating tool ──
    print(f"\n[Step 2] Generating policies for {len(mutating_tools)} tools...")

    for tool_name in mutating_tools:
        print(f"  Generating: {tool_name}...")
        tool_prompt = build_tool_policy_prompt(tool_name, wiki_text, tool_signatures)
        all_prompts.append((f"Step 2: {tool_name}", tool_prompt))

        tool_response, usage = call_llm(
            client, args.model, args.temperature, SYSTEM_PROMPT, tool_prompt
        )
        track_usage(usage)

        yaml_content = extract_yaml_block(tool_response)
        validate_policy(yaml_content, tool_name)
        filepath = output_dir / f"{tool_name}.yaml"
        filepath.write_text(yaml_content + "\n")
        print(f"    Written: {tool_name}.yaml")

    # ── Step 3: Generate general rules ──
    print("\n[Step 3] Generating general_rules.yaml...")
    general_prompt = build_general_rules_prompt(
        wiki_text, tool_signatures, mutating_tools
    )
    all_prompts.append(("Step 3: general_rules", general_prompt))

    general_response, usage = call_llm(
        client, args.model, args.temperature, SYSTEM_PROMPT, general_prompt
    )
    track_usage(usage)

    general_rules_content = extract_yaml_block(general_response)
    validate_policy(general_rules_content)
    (output_dir / "general_rules.yaml").write_text(general_rules_content + "\n")
    print("    Written: general_rules.yaml")

    # ── Step 4: Review for completeness ──
    print("\n[Step 4] Reviewing all files for completeness against wiki...")

    # Load all tool policies generated in Step 2
    tool_policies = {}
    for tool_name in mutating_tools:
        filepath = output_dir / f"{tool_name}.yaml"
        tool_policies[tool_name] = filepath.read_text().strip()

    review_prompt = build_review_prompt(
        wiki_text, general_rules_content, tool_policies
    )
    all_prompts.append(("Step 4: Review", review_prompt))

    review_response, usage = call_llm(
        client, args.model, args.temperature, SYSTEM_PROMPT, review_prompt
    )
    track_usage(usage)

    reviewed_files = parse_review_output(review_response)
    expected = {f"{tool}.yaml" for tool in mutating_tools} | {"general_rules.yaml"}
    if set(reviewed_files) != expected:
        raise ValueError("Review must return every generated policy file exactly once")
    for filename, content in reviewed_files.items():
        validate_policy(content, None if filename == "general_rules.yaml" else filename[:-5])
    for filename, content in reviewed_files.items():
        (output_dir / filename).write_text(content + "\n")
        print(f"    Reviewed: {filename}")

    # ── Summary ──
    total_calls = 1 + len(mutating_tools) + 1 + 1
    print(f"\nDone. {total_calls} LLM calls total (1 classify + {len(mutating_tools)} tools + 1 general + 1 review)")
    print(f"Tokens: {total_usage['prompt_tokens']} in / {total_usage['completion_tokens']} out / {total_usage['total_tokens']} total")
    print(f"Output: {output_dir}")

    # Save prompts log for reproducibility
    prompts_log = ""
    for title, prompt in all_prompts:
        prompts_log += f"# {title}\n\n## System Prompt\n\n{SYSTEM_PROMPT}\n\n## User Prompt\n\n{prompt}\n\n---\n\n"
    (output_dir / "_prompts.md").write_text(prompts_log)


if __name__ == "__main__":
    main()
