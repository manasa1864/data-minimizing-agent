# Minimal by Training: can fine-tuning teach AI agents to share less personal data?

By Manasa Gopisetty (solo submission), Lexsi AI Safety and Alignment Hackathon, Track 3 (Agentic Fine-tuning)

AI agents that call tools often hand over more personal data than the task needs. A delivery check also gets a phone number, and a bill payment also gets an email address. Each extra detail ends up in a third party's logs, and nobody can take it back. Earlier work measures this and mostly fixes it at inference time, by rewriting the calls after the model produces them. I asked a different question: does the style of the fine-tuning examples decide whether an agent over-shares, and can fine-tuning alone fix it?

## What I did

I used Cohere's Tiny Aya Global (3.35B parameters) and AgentTune for fine-tuning. I built a synthetic benchmark with 11 tools and fake personas. Nine tools were used for training and in-distribution testing, and two tools were held out for testing only. I compared four versions of the model, all with the same prompts:

1. The untouched model.
2. The untouched model plus a privacy rule in the prompt.
3. A fine-tune on examples that also add unneeded personal data to each call. I built this one on purpose, so its result is expected.
4. A fine-tune on the same tasks, with only the needed fields in each call.

I scored everything with AuditKIT. A "leak" is any personal field in a tool call that the tool does not need for that request. The headline number is "clean success": the call is correct and contains no unneeded personal field.

## Results

| | Untouched | Minimal fine-tune |
|---|---|---|
| Clean success, familiar tools (200 tasks) | 67.0% | 90.5% |
| Clean success, held-out tools (100 tasks) | 72.0% | 88.0% |
| Calls with a leak, familiar tools | 24.5% | 3.0% |
| Calls with a leak, held-out tools | 11.0% | 5.0% (not significant) |

On familiar tools, the minimal fine-tune removed almost all the leaks, and task success went up too, so privacy did not cost accuracy. The fine-tune trained on over-sharing examples leaked in every call, including on tools it had never seen. So the behavior the model copies comes from its training examples, and it carries over to new tools.

Two results surprised me. The prompt-only privacy rule did not help. It made leaks worse on familiar tools, from 24.5% to 42.5%. I did not test why, and my untested guess is that naming personal fields in the instruction primes the model to use them. On the held-out tools, the drop in leaks is not statistically significant. All five remaining leaks came from the model calling a familiar tool in place of the new one, and tool accuracy there fell from 99% to 95%. For new tools, the remaining problem is choosing the right tool, not over-sharing inside a correct call.

## Limits

The leak detector is a string match against the fake personas, so it misses paraphrased leaks and sometimes flags harmless text. The tasks are synthetic and templated, with single-step calls only. Each arm had one training run, and the test sets are small, so the confidence intervals cover test-set sampling only. The over-sharing arm was built by rule, not taken from real data. I did not run general-capability or safety benchmarks after fine-tuning.

## Code, data and models

The full method, tables, confidence intervals and caveats are in [`WRITEUP.md`](https://github.com/manasa1864/data-minimizing-agent/blob/main/WRITEUP.md). Everything is public:

- Code, results and write-up: https://github.com/manasa1864/data-minimizing-agent
- Models: https://huggingface.co/collections/Mango-1864/minimal-by-training-models
- Datasets: https://huggingface.co/collections/Mango-1864/minimal-by-training-datasets

I published only the minimal adapter. The over-sharing adapter stays unpublished, because it is trained to leak.

## AI assistance

Most of the code and the first draft of this article were generated with Claude (Anthropic) working from my instructions. I chose the question and direction, ran the experiments, checked the outputs and numbers against the result files, and edited the text.
