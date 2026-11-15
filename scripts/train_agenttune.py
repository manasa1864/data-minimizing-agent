#!/usr/bin/env python3
"""Fine-tune a small model on the benchmark's tool-call demonstrations with AgentTune.

Each training sample becomes an AgentTune EventLog (prompt -> demonstrated tool call) and is
trained through `agenttune.create_distill_trainer` (LoRA + TRL SFT under the hood).

Two arms of the experiment differ ONLY in `--leak-frac`:
  --leak-frac 0.0   minimal demonstrations: the call carries only the data the tool needs
  --leak-frac 1.0   overfilled demonstrations: same call, plus unneeded personal data in
                    optional arguments (what a "fill every field" teacher produces)
Values in between give a dose-response curve.

  python scripts/train_agenttune.py --leak-frac 0.0 --out runs/minimal
  python scripts/train_agenttune.py --leak-frac 1.0 --out runs/overfilled
  python scripts/train_agenttune.py --leak-frac 0.0 --out runs/x --preview   # no GPU: show rows, exit
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmin.agenttune_bridge import check_rows, to_eventlogs  # noqa: E402
from dmin.dataset import training_targets  # noqa: E402


def pick_precision(choice: str):
    """Weights dtype + the matching trainer flags, set explicitly so nothing is left to defaults.

    auto: bf16 on GPUs with native bf16 (L4, A100, ...); float16 weights (LoRA adapters stay float32)
    with fp16 autocast on GPUs without it (Colab's T4); plain float32 on CPU. A 3B+ model does not fit
    a T4 in float32.
    """
    import torch

    cuda = torch.cuda.is_available()
    if choice == "auto":
        if not cuda:
            choice = "float32"
        else:
            try:
                native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
            except TypeError:  # older torch: no such argument
                native_bf16 = torch.cuda.is_bf16_supported()
            if not native_bf16:
                return torch.float16, {"bf16": False, "fp16": True}
            choice = "bfloat16"
    flags = {"bf16": cuda and choice == "bfloat16", "fp16": cuda and choice == "float16"}
    if not cuda:
        flags["use_cpu"] = True
    return getattr(torch, choice), flags


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global", help="Hugging Face id or local path of the student")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--leak-frac", type=float, default=0.0, help="share of demonstrations that are overfilled (0..1)")
    ap.add_argument("--out", required=True, help="folder for the LoRA adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (overrides --epochs)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=1536, help="token limit per example; prompts are ~700-900 tokens")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="turn off gradient checkpointing (uses more memory, runs faster)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-targets", help="comma-separated module names (default: AgentTune's q_proj,k_proj,v_proj,o_proj)")
    ap.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    ap.add_argument("--limit", type=int, help="use only the first N training samples")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preview", action="store_true", help="build and check the training rows, then exit (no model load)")
    a = ap.parse_args()

    samples = [json.loads(line) for line in open(a.data, encoding="utf-8") if line.strip()]
    if a.limit:
        samples = samples[:a.limit]
    targets = training_targets(samples, a.leak_frac, a.seed)
    n_leaky = sum(t != s["target"] for t, s in zip(targets, samples))
    print(f"{len(samples)} demonstrations, {n_leaky} overfilled (leak_frac={a.leak_frac})")

    import agenttune
    from agenttune import create_distill_trainer

    dtype, precision = pick_precision(a.dtype)
    print(f"precision: weights {str(dtype).replace('torch.', '')}, trainer flags {precision}")

    sft_kwargs = dict(num_train_epochs=a.epochs, learning_rate=a.lr,
                      per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
                      max_length=a.max_length, logging_steps=10, save_strategy="no", report_to="none", seed=a.seed,
                      **precision)
    if not a.no_grad_ckpt:
        sft_kwargs.update(gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.max_steps > 0:
        sft_kwargs["max_steps"] = a.max_steps
    trainer = create_distill_trainer(
        student=a.model,
        trajectories=to_eventlogs(samples, targets),
        output_dir=a.out,
        lora_r=a.lora_r, lora_alpha=2 * a.lora_r,
        lora_targets=a.lora_targets.split(",") if a.lora_targets else None,
        dtype=dtype,
        **sft_kwargs,
    )

    rows = trainer.sft_dataset()  # what AgentTune will actually train on
    check_rows(rows, samples, targets)
    print(f"AgentTune built {len(rows)} training rows; they match the dataset. Last one ends with:\n"
          f"  ...{rows[-1]['messages'][0]['content'][-90:]!r}\n  -> {rows[-1]['messages'][1]['content']}")
    if a.preview:
        return

    t0 = time.time()
    result = trainer.train()
    factory = trainer.trainer_factory
    factory.sft_trainer.save_model(a.out)       # the LoRA adapter (adapter_config.json + weights)
    factory.tokenizer.save_pretrained(a.out)
    info = {"base_model": a.model, "leak_frac": a.leak_frac, "n_demonstrations": len(samples),
            "n_overfilled": n_leaky, "agenttune_version": getattr(agenttune, "__version__", "unknown"),
            "train_seconds": round(time.time() - t0, 1), "args": vars(a),
            "train_loss": getattr(result, "training_loss", None)}
    Path(a.out, "dmin_run_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"\ndone in {info['train_seconds']}s, train_loss={info['train_loss']}. adapter saved to {a.out}/")
    print(f"evaluate it with:  python scripts/eval_auditkit.py --model hf:{a.out} --data data/test_id.jsonl ...")


if __name__ == "__main__":
    main()
