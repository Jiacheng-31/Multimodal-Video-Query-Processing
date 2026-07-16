from __future__ import annotations

import re
from argparse import Namespace

import torch
from torch import nn

from .common import ACTIONS, QueryEpisode, TrajectoryState
from .prompting import remaining_budget, route_token_cost


FEATURE_DIM = 22


class ActionValueCritic(nn.Module):
    def __init__(self, input_dim: int = FEATURE_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(ACTIONS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _overlap(query_tokens: set[str], text: str) -> float:
    text_tokens = _tokens(text)
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / max(1, len(query_tokens))


def _caption_text(cap: dict) -> str:
    return str(cap.get("caption") or "")


def _candidate_preview(candidate) -> str:
    if candidate.preview_text:
        return candidate.preview_text
    return " ".join(_caption_text(cap) for cap in candidate.captions[:8])


def feature_vector(episode: QueryEpisode, state: TrajectoryState, args: Namespace) -> list[float]:
    candidate = episode.candidates[state.candidate_index]
    route = state.routes[state.candidate_index]
    n_frames = max(1, len(candidate.captions))
    idx = min(route.frame_index, n_frames - 1)
    current = candidate.captions[idx]
    query_tokens = _tokens(episode.query)
    overlaps = [_overlap(query_tokens, _caption_text(cap)) for cap in candidate.captions]
    past = overlaps[:idx]
    future = overlaps[idx + 1 :]
    current_overlap = overlaps[idx] if overlaps else 0.0
    max_future = max(future) if future else 0.0
    mean_future = sum(future) / max(1, len(future)) if future else 0.0
    max_past = max(past) if past else 0.0

    future_candidates = episode.candidates[state.candidate_index + 1 :]
    future_candidate_overlaps = [
        _overlap(query_tokens, _candidate_preview(candidate_item))
        for candidate_item in future_candidates
    ]
    max_future_candidate = max(future_candidate_overlaps) if future_candidate_overlaps else 0.0
    mean_future_candidate = (
        sum(future_candidate_overlaps) / len(future_candidate_overlaps)
        if future_candidate_overlaps
        else 0.0
    )

    global_budget = max(1.0, float(episode.route_budget))
    duration = max(1.0, float(candidate.duration))
    caption_len = len(_tokens(_caption_text(current)))
    query_len = len(query_tokens)
    frames_remaining = max(0, n_frames - idx - 1)
    candidates_total = max(1, len(episode.candidates))
    candidates_remaining = max(0, len(episode.candidates) - state.candidate_index - 1)
    score = 0.0 if candidate.retrieval_score is None else max(0.0, min(1.0, float(candidate.retrieval_score)))

    return [
        state.candidate_index / max(1, candidates_total - 1),
        idx / max(1, n_frames - 1),
        remaining_budget(episode, state, args) / global_budget,
        route_token_cost(route, args) / global_budget,
        len(route.text) / n_frames,
        len(route.visual) / n_frames,
        len(route.dropped) / n_frames,
        float(current.get("timestamp", 0.0)) / duration,
        current_overlap,
        max_future,
        mean_future,
        max_past,
        max_future_candidate,
        mean_future_candidate,
        (candidate.retrieval_rank - 1) / max(1, candidate.retrieval_total - 1),
        score,
        min(1.0, float(args.text_token_cost) / max(1.0, float(args.visual_token_cost))),
        min(1.0, float(args.visual_token_cost) / global_budget),
        candidates_remaining / candidates_total,
        frames_remaining / n_frames,
        state.step_index / max(1, sum(len(item.captions) for item in episode.candidates)),
        min(1.0, (query_len + caption_len) / 120.0),
    ]
