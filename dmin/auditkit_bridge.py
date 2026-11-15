"""Run the benchmark through AuditKIT.

`to_ak_samples` turns benchmark samples into `auditkit.Sample`s (the prompt is the input, so the
`hf:` backend sends it through the model's chat template as one user message). `make_scorers`
wraps this benchmark's scorer as AuditKIT custom scorers, each with a declared direction.
"""
from __future__ import annotations

from .dataset import prompt_for
from .scoring import score_call


def to_ak_samples(samples: list, privacy: bool = False) -> list:
    import auditkit as ak

    return [ak.Sample(input=prompt_for(s, privacy), target=s["target"], id=s["id"],
                      metadata={"tool": s["tool"], "variant": s["variant"],
                                "necessary_fields": list(s["necessary_fields"])})
            for s in samples]


def make_scorers(samples: list) -> list:
    """Five custom scorers. They look the benchmark record up by sample id."""
    import auditkit as ak

    by_id = {s["id"]: s for s in samples}
    cache: dict = {}

    def rec(sample, output):
        key = (sample.id, output)
        if key not in cache:
            cache.clear()  # scorers are called one after another on the same (sample, output)
            cache[key] = score_call(by_id[sample.id], output)
        return cache[key]

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="dm_clean_success")
    def clean_success(sample, output):
        return float(rec(sample, output)["clean_success"])

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="dm_task_success")
    def task_success(sample, output):
        return float(rec(sample, output)["success"])

    @ak.scorer(direction=ak.Direction.MINIMIZE, name="dm_any_leak")
    def any_leak(sample, output):
        return float(rec(sample, output)["n_leaked"] > 0)

    @ak.scorer(direction=ak.Direction.MINIMIZE, name="dm_leaked_fields")
    def leaked_fields(sample, output):
        return float(rec(sample, output)["n_leaked"])

    @ak.scorer(direction=ak.Direction.MAXIMIZE, name="dm_parsed")
    def parsed(sample, output):
        return float(rec(sample, output)["parsed"])

    return [clean_success, task_success, any_leak, leaked_fields, parsed]
