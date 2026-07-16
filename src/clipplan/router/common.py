from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ACTIONS = ("DROP", "TEXT", "VISUAL")
ACTION_TO_ID = {action: idx for idx, action in enumerate(ACTIONS)}
ID_TO_ACTION = {idx: action for action, idx in ACTION_TO_ID.items()}


@dataclass(frozen=True)
class GroundTruthClip:
    video_name: str
    start: float
    end: float
    relevance: float


@dataclass
class CandidateVideo:
    video_name: str
    captions: list[dict[str, Any]]
    frame_dir: Path
    duration: float
    timestamps: list[float]
    retrieval_rank: int
    retrieval_total: int
    retrieval_score: float | None = None
    preview_text: str = ""


@dataclass
class QueryEpisode:
    query_id: int
    query: str
    candidates: list[CandidateVideo]
    ground_truth: list[GroundTruthClip]
    route_budget: int
    dataset_root: Path
    dataset_name: str = "qvh"


@dataclass
class CandidateRoute:
    frame_index: int = 0
    text: list[float] = field(default_factory=list)
    visual: list[float] = field(default_factory=list)
    dropped: list[float] = field(default_factory=list)

    def copy(self) -> "CandidateRoute":
        return CandidateRoute(
            frame_index=self.frame_index,
            text=list(self.text),
            visual=list(self.visual),
            dropped=list(self.dropped),
        )


@dataclass
class TrajectoryState:
    candidate_index: int = 0
    routes: list[CandidateRoute] = field(default_factory=list)
    step_index: int = 0

    def copy(self) -> "TrajectoryState":
        return TrajectoryState(
            candidate_index=self.candidate_index,
            routes=[route.copy() for route in self.routes],
            step_index=self.step_index,
        )


@dataclass
class StepRecord:
    episode: QueryEpisode
    state: TrajectoryState
    messages: list[dict[str, Any]]
    features: list[float]
    feasible_actions: list[str]
    old_logprobs: dict[str, float]
    old_probs: dict[str, float]
    action: str
    old_action_logprob: float
    selected_cf: bool
    entropy: float
    cf_values: dict[str, float] | None = None


@dataclass
class CFExample:
    features: list[float]
    action: str
    target: float


@dataclass
class PPOExample:
    messages: list[dict[str, Any]]
    feasible_actions: list[str]
    action: str
    old_action_logprob: float
    advantage: float
