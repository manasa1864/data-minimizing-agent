"""Turn benchmark samples into AgentTune trajectories.

One sample becomes one full-tier EventLog:
    OBSERVATION  the prompt (tools + profile + request)   -> user message, not trained on
    TOOL_CALL    the demonstrated call                     -> assistant message, trained on

AgentTune's `EventLog.as_dataset_rows("sft")` then renders the call as
`<tool_call>{"name": ..., "arguments": ...}</tool_call>`, which is exactly `sample["target"]`.
"""
from __future__ import annotations

from .scoring import parse_call


def to_eventlogs(samples: list, targets: list) -> list:
    from agenttune.agentic.events import Event, EventKind, EventLog

    logs = []
    for sample, target in zip(samples, targets, strict=True):
        call = parse_call(target)
        log = EventLog(tier="full")
        log.append(Event(EventKind.OBSERVATION, {"text": sample["prompt"]}))
        log.append(Event(EventKind.TOOL_CALL, {"action": {"name": call["tool"], "arguments": call["arguments"]}}))
        logs.append(log)
    return logs


def check_rows(rows: list, samples: list, targets: list) -> None:
    """Fail loudly if AgentTune's training rows are not the prompts and calls we meant to teach."""
    if len(rows) != len(samples):
        raise RuntimeError(f"AgentTune produced {len(rows)} rows for {len(samples)} samples")
    for row, sample, target in zip(rows, samples, targets):
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        if roles != ["user", "assistant"] or msgs[0]["content"] != sample["prompt"] or msgs[1]["content"] != target:
            raise RuntimeError(f"training row for {sample['id']} does not match the dataset:\n{msgs}")
