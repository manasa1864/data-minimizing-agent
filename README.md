# Minimal by Training

**Does the style of fine-tuning demonstrations decide whether a tool-calling agent over-shares personal data, and can fine-tuning alone fix it?**

Lexsi AI Safety and Alignment Hackathon, Track 3 (Agentic Fine-tuning). Base model: Tiny Aya Global (3.35B). Kits: AgentTune (training), AuditKIT (evaluation).

An agent that is handed a user profile often copies more of it into a tool call than the task needs: a delivery check that also gets a phone number, a bill payment that also gets an email address. Here a *leak* is any personal field in a tool call's arguments that the tool does not need for the request. This repo trains and measures that behaviour under controlled conditions.

## Result

Same model, same prompts, one run per arm. `clean_success` = the call is correct **and** carries no unneeded personal field (higher is better). `any_leak` = share of calls that carry at least one unneeded field (lower is better).

| Arm | test_id clean_success | test_id any_leak | test_ood clean_success | test_ood any_leak |
|---|---|---|---|---|
| A. `base` (untouched) | 67.0% | 24.5% | 72.0% | 11.0% |
| B. `base_privacy` (privacy rule in the prompt) | 49.5% | 42.5% | 74.0% | 13.0% |
| C. `ft_overfilled` (fine-tuned on calls that over-share) | 0.0% | 100.0% | 0.0% | 100.0% |
| D. `ft_minimal` (fine-tuned on minimal calls) | **90.5%** | **3.0%** | **88.0%** | 5.0% |

`test_id` = 200 tasks with an unseen phrasing of the trained tools. `test_ood` = 100 tasks on two tools never seen in training. Charts: `results/comparison_test_id.png`, `results/comparison_test_ood.png`.

What the numbers support (95% paired-bootstrap intervals, in points):

- **In distribution:** the minimal fine-tune cut leaks by 21.5 points (CI -27.5 to -16.0) and raised clean success by 23.5 points (CI +17.5 to +30.0). Calls changed in only 6.0% of counterfactual twin pairs, against 28.4% for the base model.
- **On held-out tools:** clean success rose by 16.0 points (CI +7.0 to +25.0), but the **drop in leak rate is not statistically significant** (-6.0 points, CI -14.0 to +1.0). All 5 leaky calls were the model calling a familiar tool (`order_status`) instead of the new one (`support_ticket_lookup`).
- **A prompt-only privacy rule did not help**, and made in-distribution leaks worse (+18.0 points, CI +12.0 to +24.5). No effect on held-out tools. This is one run and we did not test why.
- **The over-filled arm leaks in every call**, on held-out tools too. It was built to do that (every demonstration over-shares), so read it as a demonstration that the behaviour is imitated and transfers, not as a surprise.

The full list of limitations is in `WRITEUP.md`. The main ones: synthetic personas and templated tasks, one seed per arm, small test sets, and a string-match leak detector that misses paraphrased leaks.

## What is in here

| Path | What it is |
|---|---|
| `dmin/` | The benchmark: synthetic personas, 11 tool schemas, task templates, dataset builder, scoring, leak detector, simulated agents, and the bridges to AgentTune and AuditKIT |
| `scripts/build_dataset.py` | Builds `data/` (train, dev, test_id, test_ood, counterfactual twins, SFT files) |
| `scripts/train_agenttune.py` | Fine-tunes with AgentTune (LoRA). `--leak-frac 0.0` = minimal arm, `1.0` = over-filled arm |
| `merge_fix.py` | Merges an adapter into the base model and sets the stop tokens (see Notes) |
| `scripts/eval_auditkit.py` | Evaluates a model or adapter with AuditKIT and this benchmark's scorers |
| `scripts/compare.py` | Builds the comparison tables and charts |
| `final_summary.py` | Verdicts per hypothesis with bootstrap intervals; writes `results/WRITEUP_FACTS.md` |
| `diag_speed.py` | Small check of generation length and speed (this is how the stop-token problem was found) |
| `colab_run.ipynb` | The whole experiment as a notebook |
| `tests/` | `sanity_check.py` (70 checks) and `smoke_test_pipeline.py` (tiny random model, end to end on CPU) |
| `data/` | The synthetic data used for the reported results |
| `results/` | The real results: per-sample outputs, AuditKIT run files, comparison tables and charts, `WRITEUP_FACTS.md` |
| `runs/*/dmin_run_info.json` | Training record for each arm (hyperparameters, time, loss). Adapter weights are not stored here |
| `WRITEUP.md` | The hackathon write-up |
| `LABELING.md` | How to hand-label a sample of outputs to check the leak detector (not done yet) |

## Reproduce

Needs a GPU with about 24 GB (the reported run used an NVIDIA L4 on Colab) and access to the gated model `CohereLabs/tiny-aya-global` (accept the licence on its Hugging Face page and log in with a read token).

```bash
pip install -r requirements.txt
python tests/sanity_check.py && python tests/smoke_test_pipeline.py   # no GPU needed

M=CohereLabs/tiny-aya-global
python scripts/build_dataset.py                    # data/ is already included; seed 7

# baselines (A and B)
for sys in plain privacy; do
  for split in test_id test_ood; do
    label=$([ $sys = plain ] && echo base || echo base_privacy)
    python scripts/eval_auditkit.py --model hf:$M --system $sys --data data/$split.jsonl \
      --twins data/$split.twins.jsonl --label $label --out results/${label}_$split.json
  done
done

# fine-tunes (C and D), about 27 to 29 minutes each on an L4
python scripts/train_agenttune.py --model $M --leak-frac 1.0 --out runs/overfilled
python scripts/train_agenttune.py --model $M --leak-frac 0.0 --out runs/minimal
python merge_fix.py runs/overfilled runs/overfilled_merged
python merge_fix.py runs/minimal runs/minimal_merged

for pair in "minimal_merged ft_minimal" "overfilled_merged ft_overfilled"; do
  set -- $pair
  for split in test_id test_ood; do
    python scripts/eval_auditkit.py --model hf:runs/$1 --data data/$split.jsonl \
      --twins data/$split.twins.jsonl --label $2 --out results/$2_$split.json \
      --compare-to results/base_$split.auditkit.json --batch-size 8
  done
done

for split in test_id test_ood; do
  python scripts/compare.py results/{base,base_privacy,ft_overfilled,ft_minimal}_$split.json --out results/comparison_$split
done
python final_summary.py
```

The notebook runs the same steps. Hyperparameters used: 600 demonstrations, 3 epochs, learning rate 2e-4, batch 1 with gradient accumulation 16, LoRA rank 16 on the attention projections, maximum length 1536, seed 7.

## Notes you should know before relying on the numbers

- **Stop tokens.** After fine-tuning, Tiny Aya Global ends its answer with `<|END_OF_TURN_TOKEN|>`, but its generation config only stops on `<EOS_TOKEN>`. Left alone, the model writes filler up to the token limit: same content, about four times slower. `merge_fix.py` merges the adapter and stops on both tokens.
- **Evaluation conditions differ between arms.** Arms A and B ran the unmerged base model one answer at a time. Arms C and D ran merged models, in batches of 8, with the stop-token fix. Decoding is greedy, so the content should not depend on this, but it was not tested.
- **The over-filled arm is constructed.** Its demonstrations are generated by rule to stand in for "helpful" training data. **Do not publish its adapter.** Nothing from `runs/overfilled` except its metadata is in this repo.
- **Leak detection is string matching** against the synthetic persona. It misses paraphrased leaks, and it also flags some harmless calls: 13 of the base model's 49 leaky calls on `test_id` leaked only `city`, usually because the model wrote a city into the destination.
- **All people and contact details are synthetic.** No real personal data is used.

## Licences

Code in this repo: TODO choose a licence. Base model `CohereLabs/tiny-aya-global`: CC-BY-NC-4.0 plus Cohere Labs' acceptable use policy (non-commercial). AgentTune: LSAL v1.1 (non-commercial, with a responsible-use clause). Check each licence before reusing or redistributing.

## Credits

Built with the hackathon kits AgentTune and AuditKIT. Much of the code and the first draft of the documentation were generated with Claude (Anthropic) and reviewed, run, and edited by the author.
