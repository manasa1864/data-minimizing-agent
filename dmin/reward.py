"""A privacy-shaped reward for AgentTune's RL trainers (GRPO / RLOO). Optional, not used by the
main SFT experiment.

    reward = task_success - leak_weight * (number of unneeded personal fields sent)

AgentTune calls reward functions as fn(completions=..., prompts=..., **dataset_columns) and each
completion is either a string or a list of chat messages. Put the benchmark record in a dataset
column called `sample_json` (a JSON string) and pass `make_privacy_reward()` in `reward_funcs`.
Sweeping `leak_weight` traces the privacy/utility curve.
"""
from __future__ import annotations

import json

from .scoring import score_call


def _text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):  # chat messages: take the last assistant turn
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content") or "")
        return ""
    return str(completion)


def make_privacy_reward(leak_weight: float = 0.5, unparsed_penalty: float = -1.0):
    def privacy_reward(completions, sample_json=None, **kwargs):
        if sample_json is None:
            raise ValueError("privacy_reward needs a `sample_json` dataset column")
        rewards = []
        for completion, raw in zip(completions, sample_json):
            rec = score_call(json.loads(raw) if isinstance(raw, str) else raw, _text(completion))
            if not rec["parsed"]:
                rewards.append(unparsed_penalty)
            else:
                rewards.append(float(rec["success"]) - leak_weight * rec["n_leaked"])
        return rewards

    privacy_reward.__name__ = f"privacy_reward_w{leak_weight}"
    return privacy_reward
