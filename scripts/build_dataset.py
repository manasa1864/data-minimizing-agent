#!/usr/bin/env python3
"""Generate the dataset.

  train.jsonl, dev.jsonl            full records, each with a minimal and an overfilled target
  test_id.jsonl, test_ood.jsonl     full records (gold call + labels), used by the eval scripts
  test_id.twins.jsonl, ...          counterfactual twins of the test samples
  train.minimal.sft.jsonl           chat-format copies of the two training arms, in case you
  train.overfilled.sft.jsonl        want to inspect them or train outside AgentTune
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmin.dataset import build_split, build_twins, to_sft_record, training_targets  # noqa: E402


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--n-dev", type=int, default=60)
    ap.add_argument("--n-test-id", type=int, default=200)
    ap.add_argument("--n-test-ood", type=int, default=100)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = {"train": a.n_train, "dev": a.n_dev, "test_id": a.n_test_id, "test_ood": a.n_test_ood}
    for split, n in sizes.items():
        rows = build_split(split, n, a.seed)
        write_jsonl(out / f"{split}.jsonl", rows)
        v, t = Counter(r["variant"] for r in rows), Counter(r["tool"] for r in rows)
        print(f"{split:9s} n={n:<4d} variants={dict(v)}  tools={len(t)}")
        if split.startswith("test"):
            write_jsonl(out / f"{split}.twins.jsonl", build_twins(rows, a.seed))
        if split == "train":
            for arm, frac in (("minimal", 0.0), ("overfilled", 1.0)):
                targets = training_targets(rows, frac, a.seed)
                write_jsonl(out / f"train.{arm}.sft.jsonl", [to_sft_record(r, t) for r, t in zip(rows, targets)])
    print(f"\nwrote {len(list(out.glob('*.jsonl')))} files to {out}/")


if __name__ == "__main__":
    main()
