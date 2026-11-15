#!/usr/bin/env python3
"""Evaluate a model on the benchmark with AuditKIT.

AuditKIT runs the model and applies this benchmark's custom scorers (dm_clean_success,
dm_task_success, dm_any_leak, dm_leaked_fields, dm_parsed). The same outputs are then broken
down by task variant, checked for over-redaction, and compared with their counterfactual twins.

  # base model, no instruction / with a privacy instruction
  python scripts/eval_auditkit.py --model hf:CohereLabs/tiny-aya-global --data data/test_id.jsonl \\
         --twins data/test_id.twins.jsonl --label base --out results/base_id.json
  python scripts/eval_auditkit.py --model hf:CohereLabs/tiny-aya-global --system privacy ... --label base+privacy

  # an AgentTune adapter folder (AuditKIT loads the base model named in adapter_config.json)
  python scripts/eval_auditkit.py --model hf:runs/minimal --data data/test_id.jsonl \\
         --twins data/test_id.twins.jsonl --label ft-minimal --out results/minimal_id.json \\
         --compare-to results/base_id.auditkit.json

  # no GPU, no download: check the plumbing with a scripted agent
  python scripts/eval_auditkit.py --model sim:naive --data data/test_id.jsonl --out results/sim.json

Writes <out> (read by compare.py) and <out minus .json>.auditkit.json (AuditKIT's own run file).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmin.agents import make_agent  # noqa: E402
from dmin.auditkit_bridge import make_scorers, to_ak_samples  # noqa: E402
from dmin.dataset import prompt_for  # noqa: E402
from dmin.scoring import build_result  # noqa: E402
from run_eval import load_jsonl, print_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="AuditKIT model spec, e.g. hf:CohereLabs/tiny-aya-global, "
                                                   "hf:runs/minimal, or sim:<oracle|naive|over_redact|tuned_sim>")
    ap.add_argument("--data", required=True)
    ap.add_argument("--twins", help="twins file for the counterfactual test (doubles the number of generations)")
    ap.add_argument("--system", choices=["plain", "privacy"], default="plain")
    ap.add_argument("--label", help="name shown in tables (default: the model spec)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, help="pipeline batch size for hf: models (default: AuditKIT's)")
    ap.add_argument("--compare-to", help="an earlier *.auditkit.json; prints AuditKIT's baseline -> candidate comparison")
    a = ap.parse_args()

    import auditkit as ak

    samples = load_jsonl(a.data, a.limit)
    ids = {s["id"] for s in samples}
    twins = [t for t in load_jsonl(a.twins) if t["twin_of"] in ids] if a.twins else []
    privacy = a.system == "privacy"

    if a.model.startswith("sim:"):
        agent = make_agent(a.model[4:])
        by_prompt = {prompt_for(s, privacy): s for s in samples + twins}
        from auditkit.model import CallableModel
        model = CallableModel(lambda prompts: [agent.act(by_prompt[p]) for p in prompts], name=a.model)
    else:
        opts = {"batch_size": a.batch_size} if a.batch_size else {}
        model = ak.AutoModel.resolve(a.model, **opts)  # loaded once, reused for the twins

    cfg = ak.RunConfig(temperature=0.0, max_tokens=a.max_tokens)
    run = ak.evaluate(to_ak_samples(samples, privacy), model, make_scorers(samples), config=cfg, cache=False)
    outputs = {p.sample_id: p.raw_output for p in run.predictions}
    failed = run.failed_count
    if twins:
        twin_run = ak.evaluate(to_ak_samples(twins, privacy), model, make_scorers(twins)[-1:], config=cfg, cache=False)
        outputs.update({p.sample_id: p.raw_output for p in twin_run.predictions})
        failed += twin_run.failed_count

    label = a.label or a.model
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ak_path = out.with_name(out.stem + ".auditkit.json")
    run.save(str(ak_path))
    result = build_result(label, a.data, samples, outputs, twins,
                          extra={"model": a.model, "system": a.system, "auditkit_headline": run.headline,
                                 "auditkit_fingerprint": run.fingerprint, "failed_generations": failed})
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=== AuditKIT run ===")
    print(run.summary())
    if failed:
        print(f"\nWARNING: {failed} generations failed inside AuditKIT; they score as unparsed.")
    print("\n=== Benchmark breakdown ===")
    print_result(result)
    if a.compare_to:
        print("\n=== AuditKIT comparison (baseline -> this run) ===")
        print(ak.compare(ak.RunResult.load(a.compare_to), run).summary())
    print(f"\nsaved {out} and {ak_path}")


if __name__ == "__main__":
    main()
