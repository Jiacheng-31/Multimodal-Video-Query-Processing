from argparse import Namespace
from pathlib import Path

from clipplan.router.common import (
    ACTIONS,
    ACTION_TO_ID,
    CandidateRoute,
    CandidateVideo,
    QueryEpisode,
    TrajectoryState,
)
from clipplan.router.critic import FEATURE_DIM, ActionValueCritic
from clipplan.router.prompting import feasible_actions, remaining_budget


def test_router_action_space_and_critic_shape() -> None:
    assert ACTIONS == ("DROP", "TEXT", "VISUAL")
    assert ACTION_TO_ID == {"DROP": 0, "TEXT": 1, "VISUAL": 2}
    critic = ActionValueCritic()
    assert FEATURE_DIM == 22
    assert critic.net[0].in_features == 22
    assert critic.net[-1].out_features == 3


def test_budget_is_shared_across_candidates() -> None:
    candidates = [
        CandidateVideo(
            video_name=f"video-{index}",
            captions=[{"timestamp": 0.0, "caption": "frame"}],
            frame_dir=Path("frames") / f"video-{index}",
            duration=2.0,
            timestamps=[0.0],
            retrieval_rank=index + 1,
            retrieval_total=2,
        )
        for index in range(2)
    ]
    episode = QueryEpisode(
        query_id=1,
        query="query",
        candidates=candidates,
        ground_truth=[],
        route_budget=12,
        dataset_root=Path("data"),
    )
    state = TrajectoryState(
        candidate_index=1,
        routes=[CandidateRoute(frame_index=1, visual=[0]), CandidateRoute()],
        step_index=1,
    )
    args = Namespace(text_token_cost=4, visual_token_cost=8)

    assert remaining_budget(episode, state, args) == 4
    assert feasible_actions(episode, state, args) == ["DROP", "TEXT"]
