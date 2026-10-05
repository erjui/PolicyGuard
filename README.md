# PolicyGuard: A Dialogue-Grounded Sub-Agent Verifier for Policy Adherence in LLM Agents

[Project page](https://policyguard.github.io/) · [Paper](https://arxiv.org/abs/2606.29225)

PolicyGuard checks policy-sensitive tool calls against the full agent-visible
conversation and returns corrective feedback before a blocked call can execute.
This release contains the runtime, checklist-generation code, and a runner for
the airline, retail, and telecom domains of tau2-bench. Generated checklists are
not stored in the repository.

## Setup

Use Python 3.12 or 3.13. From this repository directory:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
git clone https://github.com/sierra-research/tau2-bench.git external/tau2-bench
git -C external/tau2-bench checkout a2c024725189473d2d7cea3a5cfdbcc67478e41f
python -m pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Other agent models can use LiteLLM provider keys;
`--user-model` defaults to `gpt-4.1`. Benchmark data is read from
`external/tau2-bench/data`; set `TAU2_DATA_DIR` to use another data directory.
The benchmark revision is pinned because these runtimes integrate with its
orchestrator and agent interfaces.

## Run

```bash
python generate_checklists.py --domain airline --model gpt-5.4 \
  --output-dir outputs/checklists/airline
python run.py --domain airline --policy hybrid_advisory \
  --policy-dir outputs/checklists/airline \
  --model gpt-5.4 --verifier-model gpt-5.4 \
  --num-trials 4 --seed 300 --max-concurrency 4
```

This is the reported PG-CHECKLIST configuration: the raw policy is authoritative
and the generated per-tool checklist is advisory. `hybrid_advisory` is the
original experiment name. `--policy raw` reproduces PG-RAW, and
`--verifier-model` defaults to `--model`.

The verifier prompt is `HYBRID_ADVISORY_PROMPT` in
`src/policyguard/prompts.py`. `run.py` constructs
`PolicyGuardHybridVerifier`; `src/policyguard/patches.py` calls its
`check_tool_call` method before a mutating tool executes; and that method formats
the prompt and sends it to the verifier model in `src/policyguard/verifier.py`.

Both prompts include the current-turn scope correction: the one-tool-at-a-time
rule is evaluated only for the assistant turn containing the proposed call;
earlier-turn violations do not block the current call. This is the retail scope
correction used by the August 2026 runs.

Use `--task-ids 8` for specific tasks, `--test-split` for the benchmark test
split, or omit both for all tasks. `--num-tasks` limits the run. Other settings
are listed by `python run.py --help`.

Results and runtime logs are written under `outputs/` and ignored by Git.
Use a separate `--output-dir` for each configuration. Reusing a nonempty
directory requires `--resume` and the same run settings.

## Generate checklists

The four-stage generator classifies tools, generates each tool's requirements,
extracts general rules, and reviews all files against the source policy. It
uses the current generation prompts, including pre-call scope and
non-strengthening instructions. Generation uses the OpenAI API and
`OPENAI_API_KEY` from `.env`.

```bash
python generate_checklists.py --domain retail --model gpt-5.4 \
  --output-dir outputs/checklists/retail
python run.py --domain retail --policy-dir outputs/checklists/retail \
  --model gpt-5.4 --num-tasks 1
```

Use `--domain airline` or `--domain telecom` for the other domains. An optional
`--wiki` supplies another raw policy document while retaining that domain's tool
inventory. Output contains per-tool YAML, `general_rules.yaml`, and generation
logs. Use a new output directory for each generation. Model output is checked
for valid YAML, required fields, tool coverage, and complete reviewed files.

## Files

- `run.py`: main entry point.
- `generate_checklists.py`: checklist generation from domain policies.
- `src/policyguard/`: verifier, prompts, policy loading, and orchestrator integration.

## Citation

```bibtex
@article{kang2026policyguard,
  title={PolicyGuard: A Dialogue-Grounded Sub-Agent Verifier for Policy Adherence in LLM Agents},
  author={Kang, Seongjae and Yu, Taehyung and Hwang, Sung Ju},
  journal={arXiv preprint arXiv:2606.29225},
  year={2026}
}
```
