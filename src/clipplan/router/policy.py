from __future__ import annotations

import math
import random
from contextlib import nullcontext
from typing import Any, Iterable

import torch
import torch.nn.functional as F

try:
    from qwen_vl_utils import process_vision_info
except Exception:  # pragma: no cover - only used in environments without qwen-vl-utils
    process_vision_info = None


def _has_vision(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"image", "video", "image_url"}:
                return True
    return False


def _processor_inputs(
    processor,
    messages_batch: list[list[dict[str, Any]]],
    texts: list[str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"text": texts, "padding": True, "return_tensors": "pt"}
    if any(_has_vision(messages) for messages in messages_batch):
        if process_vision_info is None:
            raise RuntimeError("qwen_vl_utils is required for multimodal router prompts")
        image_inputs, video_inputs = process_vision_info(messages_batch)
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs
    return processor(**kwargs)


def _full_inputs_for_actions(
    processor,
    messages: list[dict[str, Any]],
    actions: list[str],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    eos = processor.tokenizer.eos_token or ""
    messages_batch = [messages for _ in actions]
    prompt_texts = [prompt for _ in actions]
    full_texts = [prompt + action + eos for action in actions]
    prompt_inputs = _processor_inputs(processor, messages_batch, prompt_texts)
    prompt_lens = prompt_inputs["attention_mask"].sum(dim=1).to(device)
    full_inputs = _processor_inputs(processor, messages_batch, full_texts)
    full_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in full_inputs.items()}
    return full_inputs, prompt_lens


def action_sequence_logprobs(
    model,
    processor,
    messages: list[dict[str, Any]],
    actions: list[str],
    device: torch.device,
) -> torch.Tensor:
    inputs, prompt_lens = _full_inputs_for_actions(processor, messages, actions, device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].bool()

    label_positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    target_starts = torch.clamp(prompt_lens.to(labels.device).long() - 1, min=0).unsqueeze(1)
    target_mask = mask & (label_positions >= target_starts)
    values: list[torch.Tensor] = []
    for row_idx in range(labels.shape[0]):
        positions = torch.nonzero(target_mask[row_idx], as_tuple=False).flatten()
        if positions.numel() == 0:
            values.append(logits[row_idx, 0, 0] * 0.0)
            continue
        target_logits = logits[row_idx, positions, :]
        target_labels = labels[row_idx, positions]
        token_log_probs = F.log_softmax(target_logits, dim=-1)
        values.append(token_log_probs.gather(dim=-1, index=target_labels.unsqueeze(-1)).squeeze(-1).sum())
    return torch.stack(values)


def masked_action_logprobs(
    model,
    processor,
    messages: list[dict[str, Any]],
    feasible_actions: list[str],
    device: torch.device,
    *,
    grad: bool,
) -> dict[str, torch.Tensor]:
    context = nullcontext() if grad else torch.no_grad()
    with context:
        raw = action_sequence_logprobs(model, processor, messages, feasible_actions, device)
        norm = raw - torch.logsumexp(raw, dim=0)
    return {action: norm[idx] for idx, action in enumerate(feasible_actions)}


def masked_policy_values(
    model,
    processor,
    messages: list[dict[str, Any]],
    feasible_actions: list[str],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    logprob_tensors = masked_action_logprobs(
        model,
        processor,
        messages,
        feasible_actions,
        device,
        grad=False,
    )
    logprobs = {action: float(value.detach().cpu().item()) for action, value in logprob_tensors.items()}
    probs = {action: math.exp(logprob) for action, logprob in logprobs.items()}
    total = sum(probs.values())
    if total > 0:
        probs = {action: value / total for action, value in probs.items()}
    return logprobs, probs


def sample_from_probs(probs: dict[str, float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    last_action = next(iter(probs))
    for action, prob in probs.items():
        cumulative += float(prob)
        last_action = action
        if threshold <= cumulative:
            return action
    return last_action


def normalized_entropy(probs: dict[str, float]) -> float:
    if len(probs) <= 1:
        return 0.0
    entropy = 0.0
    for prob in probs.values():
        if prob > 0:
            entropy -= prob * math.log(prob)
    return entropy / math.log(len(probs))


def ensure_actions(actions: Iterable[str]) -> list[str]:
    return [str(action).upper() for action in actions]
