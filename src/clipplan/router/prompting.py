from __future__ import annotations

from argparse import Namespace
from typing import Any

from .common import ACTIONS, CandidateRoute, CandidateVideo, QueryEpisode, TrajectoryState


SYSTEM_PROMPT = (
    "You are a budget-constrained modality router for ranked video moment retrieval.\n\n"
    "You visit candidate videos in retrieval order and frames in temporal order. "
    "For the current frame, choose exactly one action for how this frame should be represented "
    "for a downstream multimodal localization model.\n\n"
    "Actions:\n"
    "- DROP: discard only the current frame. Cost 0.\n"
    "- TEXT: keep the timestamp and caption for the current frame. Cost TEXT Cost.\n"
    "- VISUAL: keep the timestamp and image for the current frame. Cost VISUAL Cost.\n\n"
    "Each candidate video has its own routing budget cap. Use the candidate-list context and "
    "the current candidate's full caption timeline to decide whether the current frame is worth "
    "spending this video's text or visual budget on. Save VISUAL for frames where image details "
    "are likely needed beyond the caption.\n\n"
    "Output exactly one of:\n"
    "DROP\n"
    "TEXT\n"
    "VISUAL"
)


def selected_token_cost(state: TrajectoryState, args: Namespace) -> int:
    total = 0
    for route in state.routes:
        total += route_token_cost(route, args)
    return total


def route_token_cost(route: CandidateRoute, args: Namespace) -> int:
    return len(route.text) * int(args.text_token_cost) + len(route.visual) * int(args.visual_token_cost)


def remaining_budget(episode: QueryEpisode, state: TrajectoryState, args: Namespace) -> int:
    return max(0, int(episode.route_budget) - selected_token_cost(state, args))


def action_token_cost(action: str, args: Namespace) -> int:
    if action == "TEXT":
        return int(args.text_token_cost)
    if action == "VISUAL":
        return int(args.visual_token_cost)
    return 0


def is_state_terminal(episode: QueryEpisode, state: TrajectoryState) -> bool:
    return state.candidate_index >= len(episode.candidates)


def current_candidate_and_route(
    episode: QueryEpisode,
    state: TrajectoryState,
) -> tuple[CandidateVideo, CandidateRoute]:
    candidate = episode.candidates[state.candidate_index]
    route = state.routes[state.candidate_index]
    return candidate, route


def feasible_actions(episode: QueryEpisode, state: TrajectoryState, args: Namespace) -> list[str]:
    if is_state_terminal(episode, state):
        return []
    candidate, route = current_candidate_and_route(episode, state)
    if route.frame_index >= len(candidate.captions):
        return []
    remaining = remaining_budget(episode, state, args)
    actions = ["DROP"]
    if remaining >= int(args.text_token_cost):
        actions.append("TEXT")
    if remaining >= int(args.visual_token_cost):
        actions.append("VISUAL")
    return [action for action in ACTIONS if action in actions]


def routed_so_far_text(candidate: CandidateVideo, route: CandidateRoute) -> str:
    text_set = set(route.text)
    visual_set = set(route.visual)
    drop_set = set(route.dropped)
    lines: list[str] = []
    for idx in range(min(route.frame_index, len(candidate.captions))):
        cap = candidate.captions[idx]
        ts = float(cap["timestamp"])
        if ts in visual_set:
            action = "VISUAL"
        elif ts in text_set:
            action = "TEXT"
        elif ts in drop_set:
            action = "DROP"
        else:
            action = "UNVISITED"
        lines.append(f"[{ts:.1f}s] {action}")
    return "\n".join(lines) if lines else "None"


def full_timeline_text(candidate: CandidateVideo, current_index: int) -> str:
    lines: list[str] = []
    for idx, cap in enumerate(candidate.captions):
        ts = float(cap["timestamp"])
        caption = str(cap.get("caption") or "[FAILED]").replace("\n", " ").strip()
        line = f'[{ts:.1f}s] "{caption}"'
        if idx == current_index:
            line = f">>> {line} <<<"
        lines.append(line)
    return "\n".join(lines)


def candidate_list_context(episode: QueryEpisode, state: TrajectoryState, args: Namespace) -> str:
    max_items = int(getattr(args, "max_candidate_context", 0) or 0)
    candidates = episode.candidates[:max_items] if max_items > 0 else episode.candidates
    lines: list[str] = []
    for idx, candidate in enumerate(candidates):
        score = "N/A" if candidate.retrieval_score is None else f"{candidate.retrieval_score:.4f}"
        prefix = "CURRENT " if idx == state.candidate_index else ""
        preview = candidate.preview_text
        if not preview and candidate.captions:
            preview = str(candidate.captions[0].get("caption") or "")
        preview = preview.replace("\n", " ").strip()
        if len(preview) > 220:
            preview = preview[:217] + "..."
        lines.append(
            f"{prefix}Rank {idx + 1}/{len(episode.candidates)} | "
            f"Video: {candidate.video_name} | Score: {score} | "
            f"Frames: {len(candidate.captions)} | Preview: {preview or 'N/A'}"
        )
    if max_items > 0 and len(episode.candidates) > max_items:
        lines.append(f"... {len(episode.candidates) - max_items} more candidates omitted from prompt context")
    return "\n".join(lines) if lines else "None"


def current_frame_image_part(candidate: CandidateVideo, route: CandidateRoute) -> dict[str, Any] | None:
    cap = candidate.captions[route.frame_index]
    frame_file = str(cap.get("frame_file") or "")
    if not frame_file:
        return None
    image_path = candidate.frame_dir / frame_file
    if not image_path.exists():
        return None
    return {"type": "image", "image": str(image_path)}


def build_router_messages(
    episode: QueryEpisode,
    state: TrajectoryState,
    args: Namespace,
) -> list[dict[str, Any]]:
    candidate, route = current_candidate_and_route(episode, state)
    current = candidate.captions[route.frame_index]
    ts = float(current["timestamp"])
    caption = str(current.get("caption") or "[FAILED]").replace("\n", " ").strip()
    feasible = feasible_actions(episode, state, args)
    score = "N/A" if candidate.retrieval_score is None else f"{candidate.retrieval_score:.6f}"
    user_prompt = (
        "Query:\n"
        f'"{episode.query}"\n\n'
        "Global Shared Budget Across Candidates:\n"
        f"Remaining: {remaining_budget(episode, state, args)}\n"
        f"Total: {episode.route_budget}\n"
        f"Spent: {selected_token_cost(state, args)}\n"
        f"TEXT Cost: {int(args.text_token_cost)}\n"
        f"VISUAL Cost: {int(args.visual_token_cost)}\n\n"
        "Candidate List Context:\n"
        f"{candidate_list_context(episode, state, args)}\n\n"
        "Current Candidate:\n"
        f"Rank: {candidate.retrieval_rank}/{candidate.retrieval_total}\n"
        f"Video: {candidate.video_name}\n"
        f"Retrieval Score: {score}\n"
        f"Duration: {candidate.duration:.1f}s\n"
        f"Frames: {len(candidate.captions)}\n\n"
        "Current Frame:\n"
        f"Index: {route.frame_index + 1}/{len(candidate.captions)}\n"
        f"Timestamp: {ts:.1f}s\n"
        f'Caption: "{caption}"\n\n'
        "Full Current Candidate Captions:\n"
        f"{full_timeline_text(candidate, route.frame_index)}\n\n"
        "Routed So Far For Current Candidate:\n"
        f"{routed_so_far_text(candidate, route)}\n\n"
        "Feasible Actions:\n"
        + "\n".join(feasible)
    )
    include_image = bool(getattr(args, "router_include_current_image", True))
    if include_image:
        image_part = current_frame_image_part(candidate, route)
        if image_part is not None:
            content: str | list[dict[str, Any]] = [
                {"type": "text", "text": user_prompt + "\n\nCurrent Frame Image:"},
                image_part,
            ]
        else:
            content = user_prompt
    else:
        content = user_prompt
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
