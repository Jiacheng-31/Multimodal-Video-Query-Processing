from clipplan.retrieval.retrieve_candidates import rrf_fuse
from clipplan.retrieval.schema import FeatureHit, FeatureMeta, VideoHit


def _hit(video_name: str, score: float, rank: int) -> VideoHit:
    meta = FeatureMeta(feature_id=rank, video_name=video_name, granularity="test")
    feature = FeatureHit(feature_id=rank, video_name=video_name, score=score, rank=rank, meta=meta)
    return VideoHit(video_name, score, rank, feature)


def test_rrf_rewards_consensus() -> None:
    rankings = {
        "context": [_hit("shared", 0.9, 1), _hit("context-only", 0.8, 2)],
        "entity": [_hit("shared", 0.7, 1)],
        "action": [_hit("action-only", 0.9, 1)],
    }
    fused = rrf_fuse(rankings, {}, top_h=3, kappa=60)
    assert fused[0].video_name == "shared"
    assert fused[0].retrieval_score > fused[1].retrieval_score
