"""Agents the plain-Python evaluator (scripts/run_eval.py) can run.

SimulatedAgent     scripted behaviour (oracle / naive / over_redact / tuned_sim). These exist to
                   validate the harness. Their numbers are NOT research results.
OpenAICompatAgent  any /v1/chat/completions endpoint (vLLM, Ollama, ...).
CallableAgent      your own function: fn(messages: list[dict]) -> str

For Hugging Face checkpoints and AgentTune adapters use scripts/eval_auditkit.py instead.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

from .dataset import SINKS, build_messages, render_target
from .scoring import detect_fields

PRESETS = {
    "oracle": {},                                              # always the minimal gold call
    "naive": {"leak_p": 0.6},                                  # over-shares personal data
    "over_redact": {"miss_p": 1.0},                            # strips even the data it needs
    "tuned_sim": {"leak_p": 0.08, "miss_p": 0.04, "wrong_p": 0.06},  # imperfect minimizer
}


class SimulatedAgent:
    def __init__(self, name, leak_p=0.0, miss_p=0.0, wrong_p=0.0, seed=0):
        self.name, self.leak_p, self.miss_p, self.wrong_p, self.seed = name, leak_p, miss_p, wrong_p, seed

    def act(self, sample):
        # Seeded on the original sample id so a twin gets the same random decisions:
        # any change between the two calls then comes from the profile, not from the dice.
        rng = random.Random(f"{self.seed}-{self.name}-{sample.get('twin_of', sample['id'])}")
        call = copy.deepcopy(sample["gold"])
        args, persona = call["arguments"], sample["persona"]
        if rng.random() < self.wrong_p:  # corrupt one required argument
            keys = [k for k in sample["exact_keys"] if k in args]
            if keys:
                args[rng.choice(keys)] = "WRONG"
        for k in list(args):  # drop personal data the task needs
            if detect_fields({k: args[k]}, persona) and rng.random() < self.miss_p:
                args[k] = "[redacted]"
        for sink, field in SINKS.items():  # over-share into optional arguments
            if sink not in args and rng.random() < self.leak_p:
                args[sink] = persona[field]
        if "notes" not in args and rng.random() < self.leak_p / 2:
            args["notes"] = f"Customer {persona['full_name']}, phone {persona['phone']}, based in {persona['city']}."
        return render_target(call["tool"], args)


class OpenAICompatAgent:
    def __init__(self, model, base_url, system_variant="plain", api_key_env="OPENAI_API_KEY",
                 max_tokens=300, temperature=0.0, timeout=120, retries=3):
        self.model, self.base_url, self.system_variant = model, base_url.rstrip("/"), system_variant
        self.api_key_env, self.max_tokens, self.temperature = api_key_env, max_tokens, temperature
        self.timeout, self.retries = timeout, retries
        self.name = f"{model}[{system_variant}]"

    def act(self, sample):
        body = json.dumps({"model": self.model, "messages": build_messages(sample, self.system_variant),
                           "temperature": self.temperature, "max_tokens": self.max_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r)["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 502, 503, 504):
                    raise
            except urllib.error.URLError as e:
                last = e
            time.sleep(2 ** attempt)
        raise RuntimeError(f"request failed after {self.retries} attempts: {last}")


class CallableAgent:
    def __init__(self, spec, system_variant="plain"):
        mod, _, fn = spec.partition(":")
        if not fn:
            raise ValueError("--callable must look like module:function")
        sys.path.insert(0, os.getcwd())
        self.fn, self.system_variant, self.name = getattr(importlib.import_module(mod), fn), system_variant, spec

    def act(self, sample):
        return self.fn(build_messages(sample, self.system_variant))


def make_agent(kind, model=None, base_url=None, callable_spec=None, system="plain", seed=0):
    if kind in PRESETS:
        return SimulatedAgent(kind, seed=seed, **PRESETS[kind])
    if kind == "openai":
        if not (model and base_url):
            raise SystemExit("--agent openai needs --model and --base-url")
        return OpenAICompatAgent(model, base_url, system_variant=system)
    if kind == "callable":
        if not callable_spec:
            raise SystemExit("--agent callable needs --callable module:function")
        return CallableAgent(callable_spec, system_variant=system)
    raise SystemExit(f"unknown agent '{kind}'. choose from {sorted(PRESETS)} + openai, callable")
