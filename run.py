"""Run PolicyGuard on the airline, retail, or telecom domain of tau2-bench."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=["airline", "retail", "telecom"], default="airline")
    parser.add_argument("--policy", choices=["hybrid_advisory", "raw"],
                        default="hybrid_advisory")
    parser.add_argument("--policy-dir", type=Path,
                        help="Generated checklist directory (required for hybrid_advisory)")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--verifier-model", help="Defaults to the agent model")
    parser.add_argument("--user-model", default="gpt-4.1")
    tasks = parser.add_mutually_exclusive_group()
    tasks.add_argument("--task-ids", nargs="+")
    tasks.add_argument("--test-split", action="store_true")
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("num_tasks", "num_trials", "max_concurrency"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    data_dir = Path(os.environ.get("TAU2_DATA_DIR", ROOT / "external/tau2-bench/data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    os.environ["TAU2_DATA_DIR"] = str(data_dir.resolve())
    if not (data_dir / "tau2/domains" / args.domain).is_dir():
        parser.error("Benchmark data not found; follow the README setup or set TAU2_DATA_DIR")
    sys.path.insert(0, str(ROOT / "src"))

    import litellm
    from tau2.agent.llm_agent import create_llm_agent
    from tau2.data_model.simulation import TextRunConfig
    from tau2.registry import registry
    from tau2.run import run_domain
    from policyguard.patches import patch_orchestrator
    from policyguard.policy_loader import get_policy_for_tool, load_structured_policy
    from policyguard.policy_text import load_domain_policy_text
    from policyguard.verifier import PolicyGuardHybridVerifier, PolicyGuardRawVerifier

    litellm.modify_params = True
    task_ids = args.task_ids
    if args.test_split:
        splits = json.loads((data_dir / "tau2/domains" / args.domain / "split_tasks.json").read_text())
        task_ids = list(splits["test"])
    raw_policy = load_domain_policy_text(args.domain, data_dir)
    verifier_model = args.verifier_model or args.model
    if args.policy == "raw":
        verifier = PolicyGuardRawVerifier(raw_policy, model=verifier_model, domain=args.domain)
    else:
        if args.policy_dir is None:
            parser.error("--policy-dir is required for --policy hybrid_advisory; "
                         "run generate_checklists.py first")
        policy_dir = args.policy_dir
        policies = load_structured_policy(str(policy_dir))
        verifier = PolicyGuardHybridVerifier(
            policies, raw_policy, model=verifier_model, domain=args.domain
        )
        missing = sorted(tool for tool in verifier.MUTATING_TOOLS if not get_policy_for_tool(policies, tool))
        if missing:
            parser.error(f"Missing checklists for mutating tools: {', '.join(missing)}")

    output_dir = (args.output_dir or ROOT / "outputs" / args.domain / args.policy).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; choose another --output-dir or use --resume")
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_orchestrator(verifier, str(output_dir / "verifier_log.jsonl"))
    if registry.get_agent_factory("policyguard_agent") is None:
        registry.register_agent_factory(create_llm_agent, "policyguard_agent")

    config = TextRunConfig(
        domain=args.domain, agent="policyguard_agent", user="user_simulator",
        llm_agent=args.model, llm_args_agent={"temperature": 0.0, "seed": args.seed},
        llm_user=args.user_model, llm_args_user={"temperature": 0.0, "seed": args.seed},
        task_ids=task_ids, num_tasks=args.num_tasks,
        num_trials=args.num_trials, seed=args.seed, max_steps=200, max_errors=10,
        max_concurrency=args.max_concurrency, log_level="WARNING", auto_resume=args.resume,
        save_to=str(output_dir / "result.json"),
    )
    run_domain(config)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
