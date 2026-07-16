from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from pathlib import Path

import torch
from clipplan.router.data import load_episodes
from clipplan.router.env import apply_action, initial_state, is_terminal, max_rollout_steps, terminal_reward
from clipplan.router.policy import masked_policy_values
from clipplan.router.prompting import build_router_messages, feasible_actions


def _router_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        max_env_steps=args.max_env_steps,
        text_token_cost=args.text_token_cost,
        visual_token_cost=args.visual_token_cost,
        score_max_clips=args.score_max_clips,
        ndcg_k=args.ndcg_k,
        iou_threshold=args.iou_threshold,
        reward_metric="ndcg",
        router_include_current_image=args.router_include_current_image,
    )


def run_episode(model, processor, episode, args: Namespace, device: torch.device) -> dict:
    state = initial_state(episode)
    decisions: list[dict] = []
    max_steps = max_rollout_steps(episode, args)
    while not is_terminal(state, episode, max_steps):
        actions = feasible_actions(episode, state, args)
        if not actions:
            break
        messages = build_router_messages(episode, state, args)
        _, probabilities = masked_policy_values(model, processor, messages, actions, device)
        action = max(actions, key=lambda item: probabilities[item])
        decisions.append(
            {
                "candidate_index": state.candidate_index,
                "action": action,
                "probability": float(probabilities[action]),
            }
        )
        state = apply_action(state, episode, action, args)

    result = terminal_reward(episode, state, args)
    return {
        "query_id": episode.query_id,
        "query": episode.query,
        "decisions": decisions,
        "clips": result.get("clips", []),
        "ndcg": result.get("ndcg", 0.0),
        "reward": result.get("reward", 0.0),
        "cache_hit": result.get("cache_hit", False),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run greedy ClipPlan router inference and Qianfan scoring.")
    parser.add_argument("--model-path", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--annotation-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--budget-ratio", type=float, default=0.40)
    parser.add_argument("--max-env-steps", type=int, default=0)
    parser.add_argument("--text-token-cost", type=int, default=32)
    parser.add_argument("--visual-token-cost", type=int, default=512)
    parser.add_argument("--score-max-clips", type=int, default=5)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--router-include-current-image", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install the 'transformers' package before running inference.") from exc
    os.environ["CLIPPLAN_DATASET_ROOT"] = str(args.dataset_root.resolve())
    episodes = load_episodes(
        args.annotation_path,
        args.dataset_root,
        limit=args.limit,
        budget_ratio=args.budget_ratio,
        visual_token_cost=args.visual_token_cost,
        budget_mode="full_visual_ratio",
        max_candidates=args.max_candidates,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    router_args = _router_args(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(run_episode(model, processor, episode, router_args, device), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
