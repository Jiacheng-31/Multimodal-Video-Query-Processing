from clipplan.router.common import GroundTruthClip
from clipplan.router.metrics import PredictedClip, ndcg_at_k, temporal_iou


def test_temporal_iou() -> None:
    assert temporal_iou(0.0, 10.0, 5.0, 15.0) == 1.0 / 3.0
    assert temporal_iou(0.0, 1.0, 2.0, 3.0) == 0.0


def test_ndcg_rewards_temporal_match() -> None:
    predictions = [PredictedClip("video", 1.0, 5.0, 0.9)]
    ground_truth = [GroundTruthClip("video", 1.0, 5.0, 3.0)]
    score, gains = ndcg_at_k(predictions, ground_truth, k=10, iou_threshold=0.5)
    assert score == 1.0
    assert gains == [3.0]
