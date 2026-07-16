#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import yaml

from .common import ACTIONS, ACTION_TO_ID, CFExample, PPOExample
from .critic import ActionValueCritic, FEATURE_DIM
from .data import load_episodes
from .env import (
    add_counterfactual_targets,
    apply_action,
    build_ppo_examples,
    collect_base_trajectory,
    initial_state,
    record_to_trace,
    terminal_reward,
)
from .policy import masked_action_logprobs
from .prompting import build_router_messages, selected_token_cost
from .logging import LocalLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ClipPlan router with counterfactual PPO.")
    parser.add_argument("--config", type=Path, help="YAML file whose keys match command-line argument names.")
    parser.add_argument("--model-path", default="models/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dataset-root", default="data/qvh")
    parser.add_argument(
        "--annotation-path",
        default="data/qvh/annotations/train_router_pool_h60_ndcg10.json",
    )
    parser.add_argument("--output-dir", default="outputs/router")
    parser.add_argument("--cache-path", default="outputs/cache/qianfan.sqlite")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=60, help="Maximum candidates per query; 0 uses every candidate in the annotation row.")
    parser.add_argument("--max-candidate-context", type=int, default=0, help="0 shows every candidate in router prompt context.")
    parser.add_argument("--total-training-steps", type=int, default=1000)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--episodes-per-step", type=int, default=1)
    parser.add_argument("--max-env-steps", type=int, default=0)
    parser.add_argument("--budget-ratio", type=float, default=0.40)
    parser.add_argument(
        "--budget-mode",
        choices=["text_plus_visual", "full_visual_ratio"],
        default="full_visual_ratio",
        help="full_visual_ratio allocates the configured fraction of Full-Visual tokens; text_plus_visual uses an explicit text-plus-visual formula.",
    )
    parser.add_argument("--budget-per-frame-text-tokens", type=int, default=8)
    parser.add_argument("--budget-visual-frame-ratio", type=float, default=0.05)
    parser.add_argument("--text-token-cost", type=int, default=32)
    parser.add_argument("--visual-token-cost", type=int, default=512)
    parser.add_argument("--lambda-cf", type=float, default=0.12)
    parser.add_argument("--cf-rollouts", type=int, default=1)
    parser.add_argument(
        "--max-cf-states-per-episode",
        type=int,
        default=0,
        help="0 keeps probability-only CF selection; positive values keep at most this many selected CF states per episode.",
    )
    parser.add_argument(
        "--max-cf-states-per-candidate",
        type=int,
        default=0,
        help="0 disables the cap; positive values keep at most this many selected CF states per candidate video.",
    )
    parser.add_argument(
        "--cf-step-stride",
        type=int,
        default=1,
        help="Only routing states with global step_index divisible by this value are eligible for CF selection.",
    )
    parser.add_argument(
        "--cf-min-step-gap",
        type=int,
        default=0,
        help="Minimum global step-index gap between selected CF states after entropy-based pruning.",
    )
    parser.add_argument(
        "--max-cf-actions-per-state",
        type=int,
        default=0,
        help="0 evaluates every feasible action; positive values evaluate at most this many actions per selected CF state.",
    )
    parser.add_argument("--reward-metric", choices=["ndcg", "iou"], default="ndcg")
    parser.add_argument(
        "--ndcg-k",
        type=int,
        default=0,
        help="0 infers retrieval.ndcg_k from the annotation and otherwise defaults to 10.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--score-max-clips", type=int, default=5)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--actor-lr", type=float, default=1e-6)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument("--critic-epochs", type=int, default=2)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--eval-before-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--baseline-episodes",
        type=int,
        default=32,
        help="Global number of sampled episodes used for step-0 base-policy evaluation.",
    )
    parser.add_argument("--baseline-output-name", default="baseline_step0.json")
    parser.add_argument("--qianfan-rpm-limit", type=int, default=900)
    parser.add_argument("--qianfan-timeout", type=int, default=30)
    parser.add_argument("--qianfan-connect-timeout", type=int, default=10)
    parser.add_argument("--qianfan-max-retries", type=int, default=1)
    parser.add_argument("--judge-max-concurrency", type=int, default=8)
    parser.add_argument("--scorer-max-tokens", type=int, default=512)
    parser.add_argument(
        "--distributed-timeout-s",
        type=int,
        default=7200,
        help="Timeout for distributed collectives. API-backed rollouts can make ranks arrive at sync points far apart.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--router-include-current-image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-dry-run-prompt", action="store_true")
    parser.add_argument("--no-final-save", action="store_true")
    parser.add_argument("--verbose-rank-logs", action="store_true")

    preliminary, _ = parser.parse_known_args()
    if preliminary.config is not None:
        with preliminary.config.open(encoding="utf-8") as handle:
            defaults = yaml.safe_load(handle) or {}
        if not isinstance(defaults, dict):
            parser.error("The training configuration must be a YAML mapping.")
        valid_destinations = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - valid_destinations)
        if unknown:
            parser.error(f"Unknown training configuration keys: {', '.join(unknown)}")
        parser.set_defaults(**defaults)
    return parser.parse_args()


def infer_ndcg_k(args: argparse.Namespace) -> int:
    if int(args.ndcg_k) > 0:
        return int(args.ndcg_k)
    annotation_path = Path(args.annotation_path)
    try:
        with annotation_path.open(encoding="utf-8") as f:
            rows = json.load(f)
        if rows:
            ndcg_k = rows[0].get("retrieval", {}).get("ndcg_k")
            if ndcg_k is not None:
                return int(ndcg_k)
    except Exception:
        pass
    return 10


def set_reward_env(args: argparse.Namespace) -> None:
    os.environ["CLIPPLAN_DATASET_ROOT"] = str(Path(args.dataset_root).resolve())
    os.environ["QIANFAN_CACHE_PATH"] = str(Path(args.cache_path).resolve())
    os.environ["CLIPPLAN_QIANFAN_RPM_LIMIT"] = str(args.qianfan_rpm_limit)
    os.environ["CLIPPLAN_QIANFAN_TIMEOUT"] = str(args.qianfan_timeout)
    os.environ["CLIPPLAN_QIANFAN_CONNECT_TIMEOUT"] = str(args.qianfan_connect_timeout)
    os.environ["CLIPPLAN_QIANFAN_MAX_RETRIES"] = str(args.qianfan_max_retries)
    os.environ["CLIPPLAN_SCORER_MAX_TOKENS"] = str(args.scorer_max_tokens)


def rank_log(accelerator: Accelerator, event: str, enabled: bool = False, **payload: Any) -> None:
    if enabled:
        print(json.dumps({"rank": accelerator.process_index, "event": event, **payload}, ensure_ascii=False), flush=True)


def allocate_train_batch_counts(remaining_counts: list[int], target_count: int) -> list[int]:
    allocations = [0 for _ in remaining_counts]
    remaining_target = max(0, int(target_count))
    while remaining_target > 0:
        active = [idx for idx, count in enumerate(remaining_counts) if allocations[idx] < count]
        if not active:
            break
        share = max(1, (remaining_target + len(active) - 1) // len(active))
        for idx in active:
            available = remaining_counts[idx] - allocations[idx]
            take = min(available, share, remaining_target)
            allocations[idx] += take
            remaining_target -= take
            if remaining_target <= 0:
                break
    return allocations


def train_critic(
    critic,
    optimizer,
    examples: list[CFExample],
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> dict[str, float]:
    local_count = torch.tensor([len(examples)], dtype=torch.long, device=accelerator.device)
    counts = accelerator.gather(local_count)
    total_count = int(counts.sum().item())
    if total_count <= 0:
        return {"critic_loss": 0.0, "critic_examples": 0.0, "critic_skipped": 1.0}

    critic.train()
    total_loss = 0.0
    for _ in range(max(1, int(args.critic_epochs))):
        optimizer.zero_grad(set_to_none=True)
        if examples:
            features = torch.tensor([ex.features for ex in examples], dtype=torch.float32, device=accelerator.device)
            actions = torch.tensor([ACTION_TO_ID[ex.action] for ex in examples], dtype=torch.long, device=accelerator.device)
            targets = torch.tensor([ex.target for ex in examples], dtype=torch.float32, device=accelerator.device)
            mask = torch.ones_like(targets)
        else:
            features = torch.zeros((1, FEATURE_DIM), dtype=torch.float32, device=accelerator.device)
            actions = torch.zeros((1,), dtype=torch.long, device=accelerator.device)
            targets = torch.zeros((1,), dtype=torch.float32, device=accelerator.device)
            mask = torch.zeros_like(targets)

        values = critic(features)
        pred = values.gather(1, actions.unsqueeze(1)).squeeze(1)
        denom = torch.clamp(mask.sum(), min=1.0)
        loss = (((pred - targets) ** 2) * mask).sum() / denom
        accelerator.backward(loss)
        optimizer.step()
        total_loss += float(loss.detach().item())

    return {
        "critic_loss": total_loss / max(1, int(args.critic_epochs)),
        "critic_examples": float(total_count),
        "critic_skipped": 0.0,
    }


def actor_example_loss(
    model,
    processor,
    example: PPOExample,
    args: argparse.Namespace,
    device: torch.device,
    normalizer: float,
) -> torch.Tensor:
    logprobs = masked_action_logprobs(
        model,
        processor,
        example.messages,
        example.feasible_actions,
        device,
        grad=True,
    )
    new_logprob = logprobs[example.action]
    old_logprob = torch.tensor(example.old_action_logprob, dtype=new_logprob.dtype, device=device)
    advantage = torch.tensor(example.advantage, dtype=new_logprob.dtype, device=device)
    ratio = torch.exp(new_logprob - old_logprob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - args.ppo_clip, 1.0 + args.ppo_clip) * advantage
    return -torch.minimum(unclipped, clipped) / normalizer


def train_actor_ppo(
    model,
    processor,
    optimizer,
    examples: list[PPOExample],
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> dict[str, float]:
    local_count = torch.tensor([len(examples)], dtype=torch.long, device=accelerator.device)
    counts_tensor = accelerator.gather(local_count)
    counts = [int(value) for value in counts_tensor.cpu().tolist()]
    if sum(counts) <= 0:
        return {"actor_loss": 0.0, "actor_updates": 0.0, "actor_examples": 0.0, "actor_skipped": 1.0}
    if accelerator.num_processes > 1 and min(counts) == 0:
        return {"actor_loss": 0.0, "actor_updates": 0.0, "actor_examples": float(sum(counts)), "actor_skipped": 1.0}

    model.train()
    target_batch_size = max(1, int(args.train_batch_size))
    cursor = 0
    total_loss = 0.0
    updates = 0
    trained_examples = 0

    while True:
        local_remaining = max(0, len(examples) - cursor)
        remaining_tensor = accelerator.gather(
            torch.tensor([local_remaining], dtype=torch.long, device=accelerator.device)
        )
        remaining_counts = [int(value) for value in remaining_tensor.cpu().tolist()]
        total_remaining = sum(remaining_counts)
        if total_remaining <= 0:
            break

        global_real_count = min(target_batch_size, total_remaining)
        allocations = allocate_train_batch_counts(remaining_counts, global_real_count)
        local_real_count = allocations[accelerator.process_index]
        real_chunk = examples[cursor : cursor + local_real_count]
        cursor += local_real_count
        if not real_chunk:
            real_chunk = [
                PPOExample(
                    messages=examples[0].messages,
                    feasible_actions=examples[0].feasible_actions,
                    action=examples[0].action,
                    old_action_logprob=examples[0].old_action_logprob,
                    advantage=0.0,
                )
            ]

        normalizer = max(1.0, global_real_count / max(1, accelerator.num_processes))
        optimizer.zero_grad(set_to_none=True)
        chunk_loss = 0.0
        for example in real_chunk:
            loss = actor_example_loss(model, processor, example, args, accelerator.device, normalizer)
            accelerator.backward(loss)
            chunk_loss += float(loss.detach().item())
        optimizer.step()

        loss_tensor = torch.tensor([chunk_loss], dtype=torch.float32, device=accelerator.device)
        total_loss += float(accelerator.gather(loss_tensor).mean().item())
        updates += 1
        trained_examples += global_real_count

    return {
        "actor_loss": total_loss / max(1, updates),
        "actor_updates": float(updates),
        "actor_examples": float(trained_examples),
        "actor_skipped": 0.0,
    }


def save_checkpoint(
    accelerator: Accelerator,
    model,
    processor,
    critic,
    output_dir: Path,
    name: str,
) -> None:
    if not accelerator.is_main_process:
        return
    save_dir = output_dir / "checkpoints" / name
    save_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(save_dir / "actor", safe_serialization=True)
    processor.save_pretrained(save_dir / "actor")
    critic_state = accelerator.unwrap_model(critic).state_dict()
    torch.save(critic_state, save_dir / "critic.pt")


def write_metric(path: Path, step: int, data: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps({"step": step, "data": data}, ensure_ascii=False) + "\n")


def gather_objects(accelerator: Accelerator, payload: Any) -> list[Any]:
    if accelerator.num_processes <= 1:
        return [payload]
    gathered: list[Any] = [None for _ in range(accelerator.num_processes)]
    torch.distributed.all_gather_object(gathered, payload)
    return gathered


def replay_final_state(episode, records, args: argparse.Namespace):
    state = initial_state(episode)
    for record in records:
        state = apply_action(state, episode, record.action, args)
    return state


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _provider_counts(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def evaluate_base_policy(
    model,
    processor,
    episodes,
    accelerator: Accelerator,
    args: argparse.Namespace,
    output_dir: Path,
    metrics_path: Path,
    console_logger,
) -> None:
    if not bool(args.eval_before_training) or int(args.baseline_episodes) <= 0:
        return

    total_eval = min(len(episodes), int(args.baseline_episodes))
    index_rng = random.Random(int(args.seed) + 9176)
    selected_indices = list(range(len(episodes)))
    index_rng.shuffle(selected_indices)
    selected_indices = sorted(selected_indices[:total_eval])
    local_indices = selected_indices[accelerator.process_index :: accelerator.num_processes]

    model.eval()
    local_items: list[dict[str, Any]] = []
    start = time.perf_counter()
    for episode_index in local_indices:
        episode = episodes[episode_index]
        rollout_rng = random.Random(int(args.seed) + 1000003 + int(episode.query_id))
        records = collect_base_trajectory(model, processor, episode, args, accelerator.device, rollout_rng)
        final_state = replay_final_state(episode, records, args)
        result = terminal_reward(episode, final_state, args)
        action_counts = {action: 0 for action in ACTIONS}
        for record in records:
            action_counts[record.action] = action_counts.get(record.action, 0) + 1
        spent_budget = selected_token_cost(final_state, args)
        providers = list(result.get("providers") or [])
        local_items.append(
            {
                "episode_index": int(episode_index),
                "query_id": int(episode.query_id),
                "dataset": episode.dataset_name,
                "candidate_count": len(episode.candidates),
                "gt_clip_count": len(episode.ground_truth),
                "reward": float(result.get("reward", 0.0)),
                "ndcg": float(result.get("ndcg", result.get("reward", 0.0))),
                "iou": float(result.get("iou", 0.0)),
                "prediction_count": int(result.get("prediction_count", 0)),
                "parse_errors": int(result.get("parse_errors", int(bool(result.get("parse_error", False))))),
                "cache_hits": int(result.get("cache_hits", int(bool(result.get("cache_hit", False))))),
                "provider_counts": _provider_counts(providers),
                "action_counts": action_counts,
                "rollout_step_count": len(records),
                "spent_budget": int(spent_budget),
                "total_budget": int(episode.route_budget),
                "budget_used_ratio": float(spent_budget) / max(1.0, float(episode.route_budget)),
            }
        )

    gathered = gather_objects(
        accelerator,
        {
            "rank": accelerator.process_index,
            "items": local_items,
            "elapsed_s": time.perf_counter() - start,
        },
    )
    if not accelerator.is_main_process:
        return

    items = [item for payload in gathered for item in payload["items"]]
    items.sort(key=lambda item: item["episode_index"])
    rewards = [float(item["reward"]) for item in items]
    ndcgs = [float(item["ndcg"]) for item in items]
    budget_ratios = [float(item["budget_used_ratio"]) for item in items]
    rollout_steps = [float(item["rollout_step_count"]) for item in items]
    predictions = [float(item["prediction_count"]) for item in items]
    action_counts = {action: 0 for action in ACTIONS}
    provider_counts: dict[str, int] = {}
    for item in items:
        for action, count in item["action_counts"].items():
            action_counts[action] = action_counts.get(action, 0) + int(count)
        for provider, count in item["provider_counts"].items():
            provider_counts[provider] = provider_counts.get(provider, 0) + int(count)

    total_candidate_evals = sum(int(item["candidate_count"]) for item in items)
    aggregate = {
        "eval/base_episode_count": float(len(items)),
        "eval/base_reward_mean": _mean(rewards),
        "eval/base_reward_std": _std(rewards),
        "eval/base_reward_min": min(rewards) if rewards else 0.0,
        "eval/base_reward_max": max(rewards) if rewards else 0.0,
        "eval/base_ndcg_mean": _mean(ndcgs),
        "eval/base_budget_used_ratio_mean": _mean(budget_ratios),
        "eval/base_rollout_steps_mean": _mean(rollout_steps),
        "eval/base_prediction_count_mean": _mean(predictions),
        "eval/base_parse_errors": float(sum(int(item["parse_errors"]) for item in items)),
        "eval/base_cache_hits": float(sum(int(item["cache_hits"]) for item in items)),
        "eval/base_candidate_evals": float(total_candidate_evals),
        "eval/base_elapsed_s": float(sum(float(payload["elapsed_s"]) for payload in gathered) / max(1, len(gathered))),
    }
    for action, count in action_counts.items():
        aggregate[f"eval/base_action_count/{action.lower()}"] = float(count)

    payload = {
        "step": 0,
        "kind": "base_policy_before_training",
        "model_path": args.model_path,
        "dataset_root": args.dataset_root,
        "annotation_path": args.annotation_path,
        "reward_metric": args.reward_metric,
        "ndcg_k": args.ndcg_k,
        "budget_ratio": args.budget_ratio,
        "text_token_cost": args.text_token_cost,
        "visual_token_cost": args.visual_token_cost,
        "selected_episode_indices": selected_indices,
        "aggregate": aggregate,
        "action_counts": action_counts,
        "provider_counts": provider_counts,
        "rank_elapsed_s": {str(payload["rank"]): payload["elapsed_s"] for payload in gathered},
        "episodes": items,
    }
    with (output_dir / args.baseline_output_name).open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if console_logger is not None:
        console_logger.log(data=aggregate, step=0)
    write_metric(metrics_path, 0, aggregate)


def dry_run(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    episodes = load_episodes(
        Path(args.annotation_path),
        dataset_root,
        limit=args.limit,
        budget_ratio=args.budget_ratio,
        visual_token_cost=args.visual_token_cost,
        budget_mode=args.budget_mode,
        budget_per_frame_text_tokens=args.budget_per_frame_text_tokens,
        budget_visual_frame_ratio=args.budget_visual_frame_ratio,
        max_candidates=args.max_candidates,
    )
    payload: dict[str, Any] = {
        "episodes": len(episodes),
        "output_dir": args.output_dir,
        "cache_path": args.cache_path,
        "lambda_cf": args.lambda_cf,
        "cf_rollouts": args.cf_rollouts,
        "max_cf_states_per_episode": args.max_cf_states_per_episode,
        "max_cf_states_per_candidate": args.max_cf_states_per_candidate,
        "cf_step_stride": args.cf_step_stride,
        "cf_min_step_gap": args.cf_min_step_gap,
        "max_cf_actions_per_state": args.max_cf_actions_per_state,
        "reward_metric": args.reward_metric,
        "ndcg_k": args.ndcg_k,
        "budget_mode": args.budget_mode,
        "budget_per_frame_text_tokens": args.budget_per_frame_text_tokens,
        "budget_visual_frame_ratio": args.budget_visual_frame_ratio,
    }
    if episodes:
        first = episodes[0]
        payload["sample_episode"] = {
            "query_id": first.query_id,
            "dataset": first.dataset_name,
            "candidate_count": len(first.candidates),
            "gt_clip_count": len(first.ground_truth),
            "route_budget": first.route_budget,
            "first_candidate": first.candidates[0].video_name,
        }
    if args.print_dry_run_prompt and episodes:
        payload["sample_messages"] = build_router_messages(episodes[0], initial_state(episodes[0]), args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        try:
            from accelerate import Accelerator, DistributedDataParallelKwargs
            from accelerate.utils import InitProcessGroupKwargs
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install ClipPlan model dependencies before training.") from exc
    args.ndcg_k = infer_ndcg_k(args)
    set_reward_env(args)
    if args.dry_run:
        dry_run(args)
        return

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=max(600, int(args.distributed_timeout_s))))
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[pg_kwargs, ddp_kwargs])
    console_logger = LocalLogger(print_to_console=True) if accelerator.is_main_process else None
    rng = random.Random(args.seed + accelerator.process_index)
    dataset_root = Path(args.dataset_root)
    episodes = load_episodes(
        Path(args.annotation_path),
        dataset_root,
        limit=args.limit,
        budget_ratio=args.budget_ratio,
        visual_token_cost=args.visual_token_cost,
        budget_mode=args.budget_mode,
        budget_per_frame_text_tokens=args.budget_per_frame_text_tokens,
        budget_visual_frame_ratio=args.budget_visual_frame_ratio,
        max_candidates=args.max_candidates,
    )
    if not episodes:
        raise RuntimeError("No router episodes loaded.")

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "traces").mkdir(parents=True, exist_ok=True)
        with (output_dir / "run_config.json").open("w") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        avg_candidates = sum(len(episode.candidates) for episode in episodes) / max(1, len(episodes))
        avg_gt = sum(len(episode.ground_truth) for episode in episodes) / max(1, len(episodes))
        print(f"Loaded router episodes: {len(episodes)}", flush=True)
        print(f"Average candidates: {avg_candidates:.2f}; average GT clips: {avg_gt:.2f}", flush=True)
        print(f"Total training steps: {args.total_training_steps}", flush=True)
    accelerator.wait_for_everyone()

    rank_log(accelerator, "load_processor_start", args.verbose_rank_logs)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    rank_log(accelerator, "load_model_start", args.verbose_rank_logs)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    critic = ActionValueCritic(input_dim=FEATURE_DIM, hidden_dim=args.critic_hidden_dim)
    actor_optimizer = torch.optim.AdamW(model.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    model, critic, actor_optimizer, critic_optimizer = accelerator.prepare(
        model,
        critic,
        actor_optimizer,
        critic_optimizer,
    )
    rank_log(accelerator, "accelerator_prepare_done", args.verbose_rank_logs)

    local_episodes = episodes[accelerator.process_index :: accelerator.num_processes] or episodes
    metrics_path = output_dir / "metrics.jsonl"
    trace_path = output_dir / "traces" / f"rank{accelerator.process_index}.jsonl"

    evaluate_base_policy(
        model,
        processor,
        episodes,
        accelerator,
        args,
        output_dir,
        metrics_path,
        console_logger,
    )
    accelerator.wait_for_everyone()

    for step in range(args.start_step + 1, args.total_training_steps + 1):
        step_start = time.perf_counter()
        records = []
        cf_examples: list[CFExample] = []
        base_terminal_results: list[dict[str, Any]] = []
        collect_time = 0.0
        cf_time = 0.0
        reward_time = 0.0

        model.eval()
        for _ in range(args.episodes_per_step):
            episode = rng.choice(local_episodes)
            collect_start = time.perf_counter()
            episode_records = collect_base_trajectory(model, processor, episode, args, accelerator.device, rng)
            collect_time += time.perf_counter() - collect_start

            cf_start = time.perf_counter()
            cf_examples.extend(
                add_counterfactual_targets(model, processor, episode_records, args, accelerator.device, rng)
            )
            cf_time += time.perf_counter() - cf_start
            records.extend(episode_records)

            reward_start = time.perf_counter()
            base_final_state = replay_final_state(episode, episode_records, args)
            base_terminal_results.append(terminal_reward(episode, base_final_state, args))
            reward_time += time.perf_counter() - reward_start

        critic_start = time.perf_counter()
        critic_metrics = train_critic(critic, critic_optimizer, cf_examples, accelerator, args)
        critic_time = time.perf_counter() - critic_start

        ppo_examples = build_ppo_examples(records, critic, args, accelerator.device)
        actor_start = time.perf_counter()
        actor_metrics = train_actor_ppo(model, processor, actor_optimizer, ppo_examples, accelerator, args)
        actor_time = time.perf_counter() - actor_start

        base_rewards = [float(result.get("reward", 0.0)) for result in base_terminal_results]
        base_ndcgs = [float(result.get("ndcg", result.get("reward", 0.0))) for result in base_terminal_results]
        base_prediction_count = sum(float(result.get("prediction_count", 0.0)) for result in base_terminal_results)
        base_parse_errors = sum(
            float(result.get("parse_errors", int(bool(result.get("parse_error", False)))))
            for result in base_terminal_results
        )
        base_cache_hits = sum(float(result.get("cache_hits", int(bool(result.get("cache_hit", False))))) for result in base_terminal_results)
        cf_targets = [float(example.target) for example in cf_examples]
        local_stats = torch.tensor(
            [
                float(len(records)),
                float(sum(1 for record in records if record.selected_cf)),
                float(len(cf_examples)),
                float(len(ppo_examples)),
                float(sum(abs(example.advantage) for example in ppo_examples)),
                float(len(base_rewards)),
                float(sum(base_rewards)),
                float(sum(value * value for value in base_rewards)),
                min(base_rewards) if base_rewards else float("inf"),
                max(base_rewards) if base_rewards else float("-inf"),
                float(sum(base_ndcgs)),
                float(base_prediction_count),
                float(base_parse_errors),
                float(base_cache_hits),
                float(sum(cf_targets)),
                float(sum(value * value for value in cf_targets)),
                min(cf_targets) if cf_targets else float("inf"),
                max(cf_targets) if cf_targets else float("-inf"),
                float(reward_time),
            ],
            dtype=torch.float32,
            device=accelerator.device,
        )
        gathered = accelerator.gather(local_stats).view(-1, 19)
        totals = gathered.sum(dim=0).cpu().tolist()
        mean_abs_adv = totals[4] / max(1.0, totals[3])
        base_reward_count = totals[5]
        base_reward_mean = totals[6] / max(1.0, base_reward_count)
        base_reward_var = max(0.0, totals[7] / max(1.0, base_reward_count) - base_reward_mean * base_reward_mean)
        base_reward_min = float(gathered[:, 8].min().item()) if base_reward_count > 0 else 0.0
        base_reward_max = float(gathered[:, 9].max().item()) if base_reward_count > 0 else 0.0
        cf_target_count = totals[2]
        cf_target_mean = totals[14] / max(1.0, cf_target_count)
        cf_target_var = max(0.0, totals[15] / max(1.0, cf_target_count) - cf_target_mean * cf_target_mean)
        cf_target_min = float(gathered[:, 16].min().item()) if cf_target_count > 0 else 0.0
        cf_target_max = float(gathered[:, 17].max().item()) if cf_target_count > 0 else 0.0
        metrics = {
            "training/global_step": step,
            "training/total_training_steps": args.total_training_steps,
            "actor/pg_loss": actor_metrics["actor_loss"],
            "actor/update_count": actor_metrics["actor_updates"],
            "actor/train_sample_count": actor_metrics["actor_examples"],
            "actor/skipped_update": actor_metrics["actor_skipped"],
            "critic/loss": critic_metrics["critic_loss"],
            "critic/examples": critic_metrics["critic_examples"],
            "critic/skipped_update": critic_metrics["critic_skipped"],
            "counterfactual/selected_steps": totals[1],
            "counterfactual/examples": totals[2],
            "counterfactual/lambda": args.lambda_cf,
            "counterfactual/rollouts": args.cf_rollouts,
            "counterfactual/max_states_per_episode": args.max_cf_states_per_episode,
            "counterfactual/max_states_per_candidate": args.max_cf_states_per_candidate,
            "counterfactual/step_stride": args.cf_step_stride,
            "counterfactual/min_step_gap": args.cf_min_step_gap,
            "counterfactual/max_actions_per_state": args.max_cf_actions_per_state,
            "rollout/step_records": totals[0],
            "rollout/ppo_examples": totals[3],
            "rollout/mean_abs_advantage": mean_abs_adv,
            "reward/base_episode_count": base_reward_count,
            "reward/base_mean": base_reward_mean,
            "reward/base_std": math.sqrt(base_reward_var),
            "reward/base_min": base_reward_min,
            "reward/base_max": base_reward_max,
            "reward/base_ndcg_mean": totals[10] / max(1.0, base_reward_count),
            "reward/base_prediction_count_mean": totals[11] / max(1.0, base_reward_count),
            "reward/base_parse_errors": totals[12],
            "reward/base_cache_hits": totals[13],
            "reward/cf_target_mean": cf_target_mean,
            "reward/cf_target_std": math.sqrt(cf_target_var),
            "reward/cf_target_min": cf_target_min,
            "reward/cf_target_max": cf_target_max,
            "timing_s/collect": collect_time,
            "timing_s/counterfactual": cf_time,
            "timing_s/reward": reward_time,
            "timing_s/reward_rank_sum": totals[18],
            "timing_s/update_critic": critic_time,
            "timing_s/update_actor": actor_time,
            "timing_s/step": time.perf_counter() - step_start,
        }

        if accelerator.is_main_process:
            console_logger.log(data=metrics, step=step)
            write_metric(metrics_path, step, metrics)

        with trace_path.open("a") as f:
            for record in records:
                trace = record_to_trace(record, args)
                trace["train_step"] = step
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

        if args.save_freq > 0 and step % args.save_freq == 0:
            accelerator.wait_for_everyone()
            save_checkpoint(accelerator, model, processor, critic, output_dir, f"step_{step}")

    accelerator.wait_for_everyone()
    if not args.no_final_save:
        save_checkpoint(accelerator, model, processor, critic, output_dir, "final")
    accelerator.wait_for_everyone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
