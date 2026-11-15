#!/usr/bin/env python3
"""Run an agent over a split with the plain-Python harness (no AuditKIT, no GPU).

Use this for the scripted agents, or for a model served behind an OpenAI-style endpoint. For
Hugging Face checkpoints and AgentTune adapters use scripts/eval_auditkit.py.

  python scripts/run_eval.py --data data/test_id.jsonl --twins data/test_id.twins.jsonl \\
         --agent naive --out results/naive_id.json
  python scripts/run_eval.py --data data/test_id.jsonl --agent openai --model my-model \\
         --base-url http://localhost:8000/v1 --system privacy --out results/base_privacy_id.json
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmin.agents import make_agent  # noqa: E402
from dmin.scoring import build_result, format_table  # noqa: E402


def load_jsonl(path, limit=None):
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def print_result(result):
    print(f"{result['label']} on {result['data']}\n")
    print(format_table({"overall": result["summary"], **{f"  {k}": v for k, v in result["by_variant"].items()}}))
    pf = result["summary"]["per_field_leak"]
    if pf:
        print("\nleaked fields (share of calls): "
              + ", ".join(f"{k} {v:.0%}" for k, v in sorted(pf.items(), key=lambda t: -t[1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--twins", help="twins file for the counterfactual test (optional)")
    ap.add_argument("--agent", required=True, help="oracle | naive | over_redact | tuned_sim | openai | callable")
    ap.add_argument("--model")
    ap.add_argument("--base-url")
    ap.add_argument("--callable", dest="callable_spec")
    ap.add_argument("--system", choices=["plain", "privacy"], default="plain",
                    help="privacy = add a data-minimization instruction (prompt-only baseline)")
    ap.add_argument("--label", help="name shown in tables (default: agent name)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    samples = load_jsonl(a.data, a.limit)
    ids = {s["id"] for s in samples}
    twins = [t for t in load_jsonl(a.twins) if t["twin_of"] in ids] if a.twins else []
    agent = make_agent(a.agent, a.model, a.base_url, a.callable_spec, a.system, a.seed)

    def run(s):
        try:
            return s["id"], agent.act(s)
        except Exception as e:  # keep going; a failed call scores as unparsed
            return s["id"], f"[agent error] {e}"

    todo = samples + twins
    if a.workers > 1:
        with ThreadPoolExecutor(a.workers) as ex:
            outputs = dict(ex.map(run, todo))
    else:
        outputs = dict(run(s) for s in todo)

    result = build_result(a.label or agent.name, a.data, samples, outputs, twins,
                          extra={"agent": a.agent, "system": a.system})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print_result(result)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
