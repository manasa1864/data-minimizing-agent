import sys, json, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
path = sys.argv[1]
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
rows = [json.loads(l) for l in open("data/test_id.jsonl")][:4]
for r in rows:
    enc = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    n_in = enc["input_ids"].shape[1]
    t = time.time()
    out = model.generate(**enc, max_new_tokens=200, do_sample=False)
    new = out[0, n_in:]
    dt = time.time() - t
    print(f"\n{r['id']}: prompt {n_in} tokens, generated {len(new)} tokens in {dt:.1f}s "
          f"({len(new)/dt:.1f} tok/s), stopped early: {len(new) < 200}")
    print(repr(tok.decode(new, skip_special_tokens=False))[:400])
