from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import read_jsonl
from .schema import FeatureHit, FeatureMeta


def l2_normalize(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return features.astype("float32")
    arr = features.astype("float32", copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms


class GranularityIndex:
    def __init__(
        self,
        *,
        name: str,
        features: np.ndarray,
        metadata: list[FeatureMeta],
        backend: str,
        hnsw_index: Any | None = None,
    ) -> None:
        self.name = name
        self.features = l2_normalize(features)
        self.metadata = metadata
        self.backend = backend
        self.hnsw_index = hnsw_index

    def search(self, query: np.ndarray, top_p: int) -> list[FeatureHit]:
        if self.features.size == 0 or top_p <= 0:
            return []
        query = l2_normalize(query.reshape(1, -1))[0]
        k = min(top_p, len(self.metadata))
        if self.backend == "hnsw":
            if self.hnsw_index is None:
                raise RuntimeError(f"HNSW index for {self.name} is not loaded")
            labels, distances = self.hnsw_index.knn_query(query, k=k)
            ids = labels[0].tolist()
            scores = (1.0 - distances[0]).tolist()
        else:
            scores_arr = self.features @ query
            if k >= scores_arr.shape[0]:
                ids = np.argsort(-scores_arr).tolist()
            else:
                ids = np.argpartition(-scores_arr, k - 1)[:k]
                ids = ids[np.argsort(-scores_arr[ids])].tolist()
            scores = [float(scores_arr[idx]) for idx in ids]
        return [
            FeatureHit(
                feature_id=int(idx),
                video_name=self.metadata[int(idx)].video_name,
                score=float(score),
                rank=rank,
                meta=self.metadata[int(idx)],
            )
            for rank, (idx, score) in enumerate(zip(ids, scores), start=1)
        ]


def build_hnsw(features: np.ndarray, output_path: Path, *, ef_construction: int, m: int) -> None:
    try:
        import hnswlib
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("hnswlib is required for --ann-backend hnsw") from exc
    features = l2_normalize(features)
    if features.size == 0:
        return
    dim = int(features.shape[1])
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=features.shape[0], ef_construction=ef_construction, M=m)
    index.add_items(features, np.arange(features.shape[0]))
    index.save_index(str(output_path))


def load_hnsw(index_path: Path, *, dim: int, ef_search: int) -> Any:
    try:
        import hnswlib
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("hnswlib is required to load HNSW indices") from exc
    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(str(index_path))
    index.set_ef(ef_search)
    return index


def load_granularity_index(index_dir: Path, name: str, *, backend: str, ef_search: int = 128) -> GranularityIndex:
    gran_dir = index_dir / name
    features = np.load(gran_dir / "features.npy", mmap_mode="r")
    metadata_rows = read_jsonl(gran_dir / "metadata.jsonl")
    metadata = [FeatureMeta.from_json(row) for row in metadata_rows]
    hnsw_index = None
    if backend == "hnsw":
        hnsw_index = load_hnsw(gran_dir / "hnsw.bin", dim=int(features.shape[1]), ef_search=ef_search)
    return GranularityIndex(name=name, features=features, metadata=metadata, backend=backend, hnsw_index=hnsw_index)


def load_manifest(index_dir: Path) -> dict[str, Any]:
    with (index_dir / "manifest.json").open(encoding="utf-8") as f:
        return json.load(f)
