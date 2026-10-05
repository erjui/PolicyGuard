"""Load structured policy YAML files."""

import os
import yaml
from typing import Dict, Optional


def load_structured_policy(policy_dir: str) -> Dict[str, dict]:
    """Load all structured policy YAML files from a directory.

    Returns dict mapping intent_name -> policy dict.
    Also includes 'general_rules' key.
    """
    policies = {}
    for filename in os.listdir(policy_dir):
        if not filename.endswith('.yaml'):
            continue
        intent_name = filename.replace('.yaml', '')
        filepath = os.path.join(policy_dir, filename)
        with open(filepath) as f:
            policies[intent_name] = yaml.safe_load(f)
    return policies


def get_policy_for_tool(policies: Dict[str, dict], tool_name: str) -> Optional[dict]:
    """Get the structured policy for a specific tool call."""
    # Direct match (tool name == intent name in most cases)
    if tool_name in policies:
        return policies[tool_name]
    # Search by tool field
    for intent_name, policy in policies.items():
        if policy.get('tool') == tool_name:
            return policy
    return None


def format_policy_for_prompt(policy: dict) -> str:
    """Format a structured policy dict as readable text for the verifier prompt."""
    return yaml.dump(policy, default_flow_style=False, sort_keys=False)
