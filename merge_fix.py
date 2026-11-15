"""Merge a LoRA adapter into its base model and make generation stop at the end-of-turn token.

usage: python merge_fix.py runs/minimal runs/minimal_merged
"""
import sys, torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

src, dst = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(src)
model = AutoPeftModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda").merge_and_unload()
stop = [tok.convert_tokens_to_ids(t) for t in ("<EOS_TOKEN>", "<|END_OF_TURN_TOKEN|>")]
print("stop token ids:", stop, "| tokenizer eos id:", tok.eos_token_id, "| unk id:", tok.unk_token_id)
assert all(isinstance(i, int) and i != tok.unk_token_id for i in stop), "a stop token was not found in the vocabulary"
model.generation_config.eos_token_id = stop
if tok.pad_token_id is not None:
    model.generation_config.pad_token_id = tok.pad_token_id
model.save_pretrained(dst)
tok.save_pretrained(dst)
print("saved", dst)
