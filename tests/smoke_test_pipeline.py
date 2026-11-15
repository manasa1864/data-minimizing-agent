#!/usr/bin/env python3
"""End-to-end plumbing test with a tiny random model. Run:  python tests/smoke_test_pipeline.py

Builds a 2-layer random Llama-style model and tokenizer on disk (no download), then runs the real
pipeline on it: AgentTune training for 3 steps -> adapter saved -> AuditKIT loads the adapter and
evaluates 4 samples and their twins. The model is random, so its outputs are nonsense and the
scores mean nothing. What this proves is that the installed AgentTune / TRL / PEFT / AuditKIT
versions accept everything the scripts pass them. Takes about a minute on CPU.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def make_tiny_model(path: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    texts = []
    for line in open(ROOT / "data" / "train.jsonl", encoding="utf-8"):
        row = json.loads(line)
        texts += [row["prompt"], row["target"]]
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    special = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]
    tok.train_from_iterator(texts[:200], trainers.BpeTrainer(vocab_size=600, special_tokens=special))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<s>",
                                   eos_token="<|im_end|>", pad_token="<pad>")
    fast.chat_template = CHATML
    fast.save_pretrained(path)
    cfg = LlamaConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
                      bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)
    LlamaForCausalLM(cfg).save_pretrained(path)


def step(name, *cmd):
    print(f"\n--- {name}\n$ python {' '.join(cmd)}")
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True)
    print("\n".join((r.stdout or "").strip().splitlines()[-14:]))
    if r.returncode != 0:
        print((r.stderr or "")[-3000:])
        sys.exit(f"FAILED at: {name}")


def main():
    with tempfile.TemporaryDirectory() as td:
        base, run, res = Path(td, "tiny-base"), Path(td, "run"), Path(td, "res.json")
        make_tiny_model(base)
        print(f"tiny random model written to {base}")
        step("AgentTune training, 3 steps", "scripts/train_agenttune.py", "--model", str(base), "--leak-frac", "0.5",
             "--limit", "12", "--max-steps", "3", "--batch-size", "2", "--grad-accum", "1", "--dtype", "float32",
             "--out", str(run))
        assert (run / "adapter_config.json").exists(), "no adapter was saved"
        step("AuditKIT evaluation of the adapter", "scripts/eval_auditkit.py", "--model", f"hf:{run}",
             "--data", "data/test_id.jsonl", "--twins", "data/test_id.twins.jsonl", "--limit", "4",
             "--max-tokens", "12", "--label", "smoke", "--out", str(res))
        summary = json.loads(res.read_text())["summary"]
        assert summary["n"] == 4 and (res.parent / "res.auditkit.json").exists()
        step("AuditKIT evaluation of the base model, privacy prompt", "scripts/eval_auditkit.py", "--model", f"hf:{base}",
             "--system", "privacy", "--data", "data/test_id.jsonl", "--limit", "4", "--max-tokens", "12",
             "--label", "smoke-base", "--out", str(Path(td, "base.json")),
             "--compare-to", str(res.parent / "res.auditkit.json"))
    print("\nSMOKE TEST PASSED: training, adapter saving, adapter loading and evaluation all run end to end.")


if __name__ == "__main__":
    main()
