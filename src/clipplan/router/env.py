from __future__ import annotations

import random
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import torch

from clipplan.api import scorer

from .common import (
    ACTION_TO_ID,
    CFExample,
    CandidateRoute,
    PPOExample,
    QueryEpisode,
    StepRecord,
    TrajectoryState,
)
from .critic import ActionValueCritic, feature_vector
from .metrics import PredictedClip, ndcg_at_k
from .policy import masked_policy_values, normalized_entropy, sample_from_probs
from .prompting import (
    build_router_messages,
    feasible_actions,
    remaining_budget,
    route_token_cost,
    selected_token_cost,
)


def initial_state(episode: QueryEpisode) -> TrajectoryState:
    return TrajectoryState(routes=[CandidateRoute() for _ in episode.candidates])


def total_frame_steps(episode: QueryEpisode) -> int:
    return sum(len(candidate.captions) for candidate in episode.candidates)


def max_rollout_steps(episode: QueryEpisode, args: Namespace) -> int:
    if int(args.max_env_steps) > 0:
        return int(args.max_env_steps)
    return total_frame_steps(episode)


def _advance_past_finished_candidates(episode: QueryEpisode, state: TrajectoryState) -> None:
    while state.candidate_index < len(episode.candidates):
        candidate = episode.candidates[state.candidate_index]
        route = state.routes[state.candidate_index]
        if route.frame_index >= len(candidate.captions):
            state.candidate_index += 1
            continue
        break


def apply_action(
    state: TrajectoryState,
    episode: QueryEpisode,
    action: str,
    args: Namespace,
) -> TrajectoryState:
    next_state = state.copy()
    _advance_past_finished_candidates(episode, next_state)
    if next_state.candidate_index >= len(episode.candidates):
        return next_state

    candidate = episode.candidates[next_state.candidate_index]
    route = next_state.routes[next_state.candidate_index]
    current = candidate.captions[route.frame_index]
    ts = float(current["timestamp"])
    feasible = feasible_actions(episode, next_state, args)
    if action not in feasible:
        action = "DROP"

    if action == "TEXT":
        route.text.append(ts)
        route.frame_index += 1
    elif action == "VISUAL":
        route.visual.append(ts)
        route.frame_index += 1
    else:
        route.dropped.append(ts)
        route.frame_index += 1

    next_state.step_index += 1
    _advance_past_finished_candidates(episode, next_state)
    return next_state


def is_terminal(
    state: TrajectoryState,
    episode: QueryEpisode,
    max_steps: int | None = None,
) -> bool:
    if state.candidate_index >= len(episode.candidates):
        return True
    if max_steps is not None and state.step_index >= max_steps:
        return True
    return False


def _score_candidate_for_ndcg(
    episode: QueryEpisode,
    state: TrajectoryState,
    candidate_idx: int,
    args: Namespace,
) -> dict[str, Any]:
    candidate = episode.candidates[candidate_idx]
    route = state.routes[candidate_idx]
    try:
        scored = scorer.score_routed_candidate(
            episode.query,
            candidate.video_name,
            candidate.duration,
            candidate.captions,
            candidate.frame_dir,
            route.text,
            route.visual,
            max_clips=int(args.score_max_clips),
        )
    except Exception as exc:
        return {
            "video_name": candidate.video_name,
            "clips": [],
            "parse_error": True,
            "cache_hit": False,
            "provider": "error",
            "error": repr(exc)[:240],
        }
    scored["video_name"] = candidate.video_name
    return scored


def _terminal_ndcg_reward(episode: QueryEpisode, state: TrajectoryState, args: Namespace) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    max_workers = max(1, int(getattr(args, "judge_max_concurrency", 1) or 1))
    candidate_indices = list(range(len(episode.candidates)))
    if max_workers > 1 and len(candidate_indices) > 1:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(candidate_indices))) as executor:
            futures = {
                executor.submit(_score_candidate_for_ndcg, episode, state, idx, args): idx
                for idx in candidate_indices
            }
            tmp: dict[int, dict[str, Any]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    tmp[idx] = future.result()
                except Exception as exc:
                    tmp[idx] = {
                        "video_name": episode.candidates[idx].video_name,
                        "clips": [],
                        "parse_error": True,
                        "cache_hit": False,
                        "provider": "error",
                        "error": repr(exc)[:240],
                    }
            results = [tmp[idx] for idx in candidate_indices]
    else:
        results = [_score_candidate_for_ndcg(episode, state, idx, args) for idx in candidate_indices]

    predictions: list[PredictedClip] = []
    for result in results:
        video_name = str(result.get("video_name") or "")
        for clip in result.get("clips", []):
            try:
                start = float(clip["start"])
                end = float(clip["end"])
                score = float(clip.get("score", 1.0))
            except (TypeError, ValueError, KeyError):
                continue
            if end > start:
                predictions.append(PredictedClip(video_name=video_name, start=start, end=end, score=score))
    ndcg, assigned = ndcg_at_k(
        predictions,
        episode.ground_truth,
        k=int(args.ndcg_k),
        iou_threshold=float(args.iou_threshold),
    )
    return {
        "reward": float(ndcg),
        "ndcg": float(ndcg),
        "assigned_relevances": assigned,
        "prediction_count": len(predictions),
        "parse_errors": sum(1 for result in results if result.get("parse_error")),
        "cache_hits": sum(1 for result in results if result.get("cache_hit")),
        "providers": [result.get("provider") for result in results],
        "outputs": [
            {
                "video_name": result.get("video_name"),
                "clips": result.get("clips", [])[: int(args.score_max_clips)],
                "parse_error": bool(result.get("parse_error", False)),
                "parse_method": result.get("parse_method"),
                "provider": result.get("provider"),
                "cache_hit": bool(result.get("cache_hit", False)),
            }
            for result in results
        ],
    }


def _terminal_iou_reward(episode: QueryEpisode, state: TrajectoryState, args: Namespace) -> dict[str, Any]:
    candidate = episode.candidates[0]
    route = state.routes[0]
    gt = episode.ground_truth[0]
    scored = scorer.score_routed_state(
        episode.query,
        candidate.video_name,
        candidate.duration,
        gt.start,
        gt.end,
        candidate.captions,
        candidate.frame_dir,
        route.text,
        route.visual,
    )
    reward = 0.0 if scored.get("parse_error") else float(scored.get("iou", 0.0))
    return {
        "reward": reward,
        "iou": float(scored.get("iou", 0.0)),
        "pred": scored.get("pred"),
        "parse_error": bool(scored.get("parse_error", False)),
        "provider": scored.get("provider"),
        "cache_hit": bool(scored.get("cache_hit", False)),
    }


def terminal_reward(episode: QueryEpisode, state: TrajectoryState, args: Namespace) -> dict[str, Any]:
    if args.reward_metric == "iou":
        return _terminal_iou_reward(episode, state, args)
    return _terminal_ndcg_reward(episode, state, args)


def rollout_policy_continuation(
    model,
    processor,
    episode: QueryEpisode,
    start_state: TrajectoryState,
    args: Namespace,
    device: torch.device,
    rng: random.Random,
    max_steps: int,
) -> TrajectoryState:
    state = start_state.copy()
    while not is_terminal(state, episode, max_steps):
        feasible = feasible_actions(episode, state, args)
        if not feasible:
            break
        messages = build_router_messages(episode, state, args)
        _, probs = masked_policy_values(model, processor, messages, feasible, device)
        action = sample_from_probs(probs, rng)
        state = apply_action(state, episode, action, args)
    return state


def _cf_state_is_eligible(record: StepRecord, args: Namespace) -> bool:
    stride = max(1, int(getattr(args, "cf_step_stride", 1) or 1))
    if stride > 1 and int(record.state.step_index) % stride != 0:
        return False
    return True


def _apply_counterfactual_selection_caps(records: list[StepRecord], args: Namespace) -> None:
    selected = [record for record in records if record.selected_cf]
    if not selected:
        return

    for record in selected:
        if not _cf_state_is_eligible(record, args):
            record.selected_cf = False

    selected = [record for record in records if record.selected_cf]
    if not selected:
        return

    max_per_episode = int(getattr(args, "max_cf_states_per_episode", 0) or 0)
    max_per_candidate = int(getattr(args, "max_cf_states_per_candidate", 0) or 0)
    min_gap = max(0, int(getattr(args, "cf_min_step_gap", 0) or 0))
    if max_per_episode <= 0 and max_per_candidate <= 0 and min_gap <= 0:
        return

    ranked = sorted(selected, key=lambda item: (-float(item.entropy), int(item.state.step_index)))
    kept: list[StepRecord] = []
    per_candidate: dict[int, int] = {}
    for record in ranked:
        if max_per_episode > 0 and len(kept) >= max_per_episode:
            record.selected_cf = False
            continue
        candidate_idx = int(record.state.candidate_index)
        if max_per_candidate > 0 and per_candidate.get(candidate_idx, 0) >= max_per_candidate:
            record.selected_cf = False
            continue
        if min_gap > 0 and any(abs(int(record.state.step_index) - int(prev.state.step_index)) < min_gap for prev in kept):
            record.selected_cf = False
            continue
        kept.append(record)
        per_candidate[candidate_idx] = per_candidate.get(candidate_idx, 0) + 1

    kept_ids = {id(record) for record in kept}
    for record in selected:
        if id(record) not in kept_ids:
            record.selected_cf = False


def collect_base_trajectory(
    model,
    processor,
    episode: QueryEpisode,
    args: Namespace,
    device: torch.device,
    rng: random.Random,
) -> list[StepRecord]:
    records: list[StepRecord] = []
    state = initial_state(episode)
    max_steps = max_rollout_steps(episode, args)

    while not is_terminal(state, episode, max_steps):
        feasible = feasible_actions(episode, state, args)
        if not feasible:
            break
        messages = build_router_messages(episode, state, args)
        old_logprobs, old_probs = masked_policy_values(model, processor, messages, feasible, device)
        entropy = normalized_entropy(old_probs)
        select_prob = float(args.lambda_cf) * (1.0 + entropy) / 2.0
        selected_cf = rng.random() < select_prob
        action = sample_from_probs(old_probs, rng)
        record = StepRecord(
            episode=episode,
            state=state.copy(),
            messages=messages,
            features=feature_vector(episode, state, args),
            feasible_actions=feasible,
            old_logprobs=old_logprobs,
            old_probs=old_probs,
            action=action,
            old_action_logprob=float(old_logprobs[action]),
            selected_cf=selected_cf,
            entropy=entropy,
        )
        records.append(record)
        state = apply_action(state, episode, action, args)

    _apply_counterfactual_selection_caps(records, args)
    return records


def estimate_counterfactual_values(
    model,
    processor,
    record: StepRecord,
    args: Namespace,
    device: torch.device,
    rng: random.Random,
) -> dict[str, float]:
    max_steps = max_rollout_steps(record.episode, args)
    rollout_count = max(1, int(args.cf_rollouts))
    seeds = [rng.randint(0, 2**31 - 1) for _ in range(rollout_count)]
    values: dict[str, float] = {}

    actions = list(record.feasible_actions)
    max_actions = int(getattr(args, "max_cf_actions_per_state", 0) or 0)
    if max_actions > 0 and max_actions < len(actions):
        selected_actions = [record.action] if record.action in actions else []
        remaining_actions = [action for action in actions if action not in selected_actions]
        rng.shuffle(remaining_actions)
        selected_actions.extend(remaining_actions[: max(0, max_actions - len(selected_actions))])
        actions = selected_actions

    for action in actions:
        returns: list[float] = []
        for seed in seeds:
            forced_state = apply_action(record.state, record.episode, action, args)
            cont_rng = random.Random(seed)
            final_state = rollout_policy_continuation(
                model,
                processor,
                record.episode,
                forced_state,
                args,
                device,
                cont_rng,
                max_steps,
            )
            result = terminal_reward(record.episode, final_state, args)
            returns.append(float(result["reward"]))
        values[action] = sum(returns) / max(1, len(returns))

    return values


def add_counterfactual_targets(
    model,
    processor,
    records: list[StepRecord],
    args: Namespace,
    device: torch.device,
    rng: random.Random,
) -> list[CFExample]:
    examples: list[CFExample] = []
    for record in records:
        if not record.selected_cf:
            continue
        record.cf_values = estimate_counterfactual_values(model, processor, record, args, device, rng)
        for action, target in record.cf_values.items():
            examples.append(CFExample(features=record.features, action=action, target=float(target)))
    return examples


def build_ppo_examples(
    records: list[StepRecord],
    critic: ActionValueCritic,
    args: Namespace,
    device: torch.device,
) -> list[PPOExample]:
    examples: list[PPOExample] = []
    critic.eval()
    for record in records:
        with torch.no_grad():
            features = torch.tensor([record.features], dtype=torch.float32, device=device)
            pred = critic(features)[0].detach().cpu().tolist()
        q_values = {action: float(pred[ACTION_TO_ID[action]]) for action in record.feasible_actions}
        if record.cf_values is not None:
            q_values.update({action: float(value) for action, value in record.cf_values.items()})

        baseline = 0.0
        for action in record.feasible_actions:
            baseline += float(record.old_probs[action]) * float(q_values[action])
        advantage = float(q_values[record.action]) - baseline
        examples.append(
            PPOExample(
                messages=record.messages,
                feasible_actions=record.feasible_actions,
                action=record.action,
                old_action_logprob=record.old_action_logprob,
                advantage=advantage,
            )
        )
    return examples


def record_to_trace(record: StepRecord, args: Namespace) -> dict[str, Any]:
    candidate = record.episode.candidates[record.state.candidate_index]
    route = record.state.routes[record.state.candidate_index]
    current = candidate.captions[route.frame_index]
    return {
        "query_id": record.episode.query_id,
        "dataset": record.episode.dataset_name,
        "candidate_index": record.state.candidate_index,
        "candidate_count": len(record.episode.candidates),
        "video_name": candidate.video_name,
        "frame_index": route.frame_index,
        "timestamp": float(current["timestamp"]),
        "caption": str(current.get("caption") or ""),
        "action": record.action,
        "selected_cf": record.selected_cf,
        "entropy": record.entropy,
        "feasible_actions": record.feasible_actions,
        "old_probs": record.old_probs,
        "cf_values": record.cf_values,
        "text": list(route.text),
        "visual": list(route.visual),
        "dropped": list(route.dropped),
        "global_step_index": record.state.step_index,
        "remaining_budget": remaining_budget(record.episode, record.state, args),
        "spent_budget": selected_token_cost(record.state, args),
        "total_budget": record.episode.route_budget,
        "current_candidate_spent": route_token_cost(route, args),
    }
