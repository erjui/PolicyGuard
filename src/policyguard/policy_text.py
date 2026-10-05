"""Domain-aware loader for raw policy text.

Returns the same text the tau2 agent sees in its system prompt:
- airline / retail: a single `policy.md` file.
- telecom: `main_policy.md` + `tech_support_manual.md` wrapped in
  `<main_policy>...` / `<tech_support_policy>...` tags, matching
  tau2/domains/telecom/environment.py.
"""

from pathlib import Path


def load_domain_policy_text(domain: str, data_dir: Path) -> str:
    base = data_dir / "tau2" / "domains" / domain
    if domain == "telecom":
        main = (base / "main_policy.md").read_text()
        tech = (base / "tech_support_manual.md").read_text()
        return (
            "<main_policy>\n"
            + main
            + "\n</main_policy>\n"
            + "<tech_support_policy>\n"
            + tech
            + "\n</tech_support_policy>"
        )
    return (base / "policy.md").read_text()
