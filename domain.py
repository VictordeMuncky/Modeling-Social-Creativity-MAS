"""
Domain Memory Abstraction
=========================

A single shared cultural-memory object for the simulation.

The Domain stores artifacts and metadata once, then supports multiple
organizational interpretations through different retrieval modes:
    - flat       : unstructured archive with random retrieval
    - similarity : feature-space neighborhood retrieval
    - lineage    : ancestry/derivation retrieval

The goal is to keep the rest of the simulation stable while making the
Domain (explicitly) an experimental variable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from framework import Artifact


class Domain:
    """
    Single cultural-memory object with multiple retrieval structures.

    Artifacts are stored once. Different domain modes change how the same
    memory is indexed and retrieved.
    """

    VALID_MODES = ("flat", "similarity", "lineage")
    VALID_STRATEGIES = ("nearest", "mid", "far")

    def __init__(self, mode: str = "flat"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported domain mode '{mode}'. Expected one of {self.VALID_MODES}.")

        self.mode = mode
        self.artifacts: List[Artifact] = []
        self.artifact_ids = set()
        self.artifact_by_id: Dict[int, Artifact] = {}
        self.by_creator = defaultdict(list)

    def __len__(self) -> int:
        return len(self.artifacts)

    def __iter__(self):
        return iter(self.artifacts)

    def __bool__(self) -> bool:
        return bool(self.artifacts)

    def contains(self, artifact_id: Optional[int]) -> bool:
        return artifact_id is not None and artifact_id in self.artifact_ids

    def get(self, artifact_id: Optional[int]) -> Optional[Artifact]:
        if artifact_id is None:
            return None
        return self.artifact_by_id.get(int(artifact_id))

    def prepare_artifact(self, artifact: Artifact, step: int, source: str = "generation"):
        """
        Populate/refresh artifact metadata before it enters the domain.
        """
        metadata = artifact.metadata
        metadata['artifact_id'] = artifact.id
        metadata['producer_id'] = artifact.producer_id
        metadata['parent_ids'] = [pid for pid in (artifact.parent1_id, artifact.parent2_id) if pid is not None]
        metadata['generation_step'] = step
        metadata['domain_source'] = source
        metadata['feature_dims'] = None if artifact.features is None else int(artifact.features.shape[0])

        inherited_path = [int(pid) for pid in metadata.get('domain_lineage_path', []) if pid is not None]
        inherited_root = metadata.get('domain_lineage_root_id')
        if inherited_path:
            metadata['domain_parent_id'] = inherited_path[-1]
            metadata['domain_lineage_root_id'] = inherited_root if inherited_root is not None else inherited_path[0]
            metadata['domain_lineage_path'] = inherited_path
            metadata['domain_lineage_depth'] = max(0, len(inherited_path) - 1)
        else:
            metadata['domain_parent_id'] = None
            metadata['domain_lineage_root_id'] = inherited_root
            metadata['domain_lineage_path'] = []
            metadata['domain_lineage_depth'] = 0

        root_creator_id = artifact.creator_id
        if inherited_path:
            domain_basis = self.get(inherited_path[-1])
            if domain_basis is not None:
                root_creator_id = domain_basis.metadata.get('root_creator_id', root_creator_id)
        else:
            parent_depths = []
            for parent_id in metadata['parent_ids']:
                parent = self.get(parent_id)
                if parent is not None:
                    root_creator_id = parent.metadata.get('root_creator_id', root_creator_id)
                    parent_depths.append(int(parent.metadata.get('lineage_depth', 0)))
            metadata['lineage_depth'] = (max(parent_depths) + 1) if parent_depths else (0 if not metadata['parent_ids'] else 1)

        if 'lineage_depth' not in metadata:
            metadata['lineage_depth'] = 0 if not metadata['parent_ids'] else 1

        metadata['root_creator_id'] = root_creator_id
        metadata['lineage_signature'] = f"{metadata['root_creator_id']}:{artifact.parent1_id}:{artifact.parent2_id}"

        artifact.refresh_popularity_score()
        self.update_similarity_metadata(artifact)
        return artifact

    def add_artifact(self, artifact: Artifact, accepted_by: int, step: int) -> Tuple[bool, float]:
        """
        Register an artifact in the shared domain memory and update indexes.

        Returns:
            (is_new_artifact, popularity_score)
        """
        self.prepare_artifact(
            artifact,
            step=artifact.metadata.get('generation_step', step) or step,
            source=artifact.metadata.get('domain_source', 'generation'),
        )

        inherited_path = list(artifact.metadata.get('domain_lineage_path', []))
        if inherited_path:
            artifact.metadata['domain_parent_id'] = inherited_path[-1]
            artifact.metadata['domain_lineage_root_id'] = artifact.metadata.get('domain_lineage_root_id', inherited_path[0])
            if inherited_path[-1] != artifact.id:
                artifact.metadata['domain_lineage_path'] = inherited_path + [artifact.id]
        else:
            artifact.metadata['domain_parent_id'] = None
            artifact.metadata['domain_lineage_root_id'] = artifact.id
            artifact.metadata['domain_lineage_path'] = [artifact.id]

        artifact.metadata['domain_lineage_depth'] = max(0, len(artifact.metadata['domain_lineage_path']) - 1)

        is_new = artifact.id not in self.artifact_ids
        if is_new:
            self.artifacts.append(artifact)
            self.artifact_ids.add(artifact.id)
            self.artifact_by_id[artifact.id] = artifact

        self.by_creator[artifact.creator_id].append(artifact.id)
        artifact.add_domain_entry(accepted_by, step)
        popularity = artifact.refresh_popularity_score()
        self.update_similarity_metadata(artifact)
        return is_new, popularity

    def random_artifact(self, exclude_artifact_id: Optional[int] = None) -> Tuple[Optional[Artifact], Dict[str, object]]:
        candidates = [a for a in self.artifacts if a.id != exclude_artifact_id]
        if not candidates:
            return None, {'retrieval_mode': 'flat', 'relation_type': 'empty', 'score': None}
        artifact = random.choice(candidates)
        return artifact, {
            'retrieval_mode': 'flat',
            'relation_type': 'random',
            'score': 1.0,
            'strategy': None,
            'strategy_value': None,
            'rank': 0,
            'pool_size': len(candidates),
            'bucket': 'random',
            'estimated_novelty': None,
        }

    def retrieve(
        self,
        query_artifact: Optional[Artifact] = None,
        query_features: Optional[torch.Tensor] = None,
        query_artifact_id: Optional[int] = None,
        mode: Optional[str] = None,
        exclude_artifact_id: Optional[int] = None,
        strategy: str = 'nearest',
        strategy_value: Optional[float] = None,
        preferred_novelty: Optional[float] = None,
        current_novelty: Optional[float] = None,
        query_metadata: Optional[Dict[str, object]] = None,
    ) -> Tuple[Optional[Artifact], Dict[str, object]]:
        """
        Retrieve one artifact according to the selected domain structure.
        """
        mode = mode or self.mode
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported domain mode '{mode}'.")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unsupported domain strategy '{strategy}'. Expected one of {self.VALID_STRATEGIES}.")
        if not self.artifacts:
            return None, {'retrieval_mode': mode, 'relation_type': 'empty', 'score': None, 'strategy': strategy, 'strategy_value': strategy_value}

        if query_artifact is not None:
            query_artifact_id = query_artifact.id
            query_features = query_artifact.features if query_features is None else query_features
            query_metadata = query_artifact.metadata if query_metadata is None else query_metadata

        exclude_id = exclude_artifact_id if exclude_artifact_id is not None else query_artifact_id

        if mode == 'flat':
            artifact, info = self.random_artifact(exclude_artifact_id=exclude_id)
            info['strategy'] = strategy
            info['strategy_value'] = strategy_value
            return artifact, info

        if mode == 'similarity':
            scored = self._scored_similarity_candidates(query_features=query_features, exclude_artifact_id=exclude_id)
            return self._select_scored_candidate(
                scored,
                retrieval_mode='similarity',
                fallback_exclude_artifact_id=exclude_id,
                strategy=strategy,
                strategy_value=strategy_value,
                preferred_novelty=preferred_novelty,
                current_novelty=current_novelty,
            )

        scored = self._scored_lineage_candidates(
            query_artifact_id=query_artifact_id,
            query_metadata=query_metadata,
            exclude_artifact_id=exclude_id,
        )
        return self._select_scored_candidate(
            scored,
            retrieval_mode='lineage',
            fallback_exclude_artifact_id=exclude_id,
            strategy=strategy,
            strategy_value=strategy_value,
            preferred_novelty=preferred_novelty,
            current_novelty=current_novelty,
        )

    def query_related(self, artifact: Artifact, k: int = 5, mode: Optional[str] = None) -> List[Tuple[Artifact, float, str]]:
        """
        Return top related artifacts according to the selected structure.
        """
        mode = mode or self.mode
        exclude_id = artifact.id
        candidates = [a for a in self.artifacts if a.id != exclude_id]
        if not candidates:
            return []

        if mode == 'flat':
            sample = random.sample(candidates, k=min(k, len(candidates)))
            return [(a, 1.0, 'random') for a in sample]

        if mode == 'similarity':
            scored = self._scored_similarity_candidates(query_features=artifact.features, exclude_artifact_id=exclude_id)
            return [(cand, score, relation_type) for cand, score, relation_type in scored[:k]]

        if mode == 'lineage':
            scored = self._scored_lineage_candidates(query_artifact_id=artifact.id, query_metadata=artifact.metadata, exclude_artifact_id=exclude_id)
            return [(cand, score, relation_type) for cand, score, relation_type in scored[:k]]

        raise ValueError(f"Unsupported domain mode '{mode}'.")

    def update_similarity_metadata(self, artifact: Artifact):
        metadata = artifact.metadata
        metadata['feature_dims'] = None if artifact.features is None else int(artifact.features.shape[0])

        scored = self._scored_similarity_candidates(query_features=artifact.features, exclude_artifact_id=artifact.id)
        if not scored:
            metadata['nearest_domain_artifact_id'] = None
            metadata['nearest_domain_similarity'] = None
            metadata['domain_cluster_hint'] = None
            return

        related, score, relation_type = scored[0]
        metadata['nearest_domain_artifact_id'] = related.id
        metadata['nearest_domain_similarity'] = score
        metadata['domain_cluster_hint'] = relation_type or f"creator_{related.creator_id}"

    def _normalized_features(self, features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if features is None:
            return None
        features = features.float()
        if features.ndim != 1:
            features = features.flatten()
        if not torch.isfinite(features).all():
            features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = torch.norm(features, p=2)
        if norm <= 1e-8:
            return None
        return F.normalize(features, dim=0)

    def _scored_similarity_candidates(
        self,
        query_features: Optional[torch.Tensor],
        exclude_artifact_id: Optional[int] = None,
    ) -> List[Tuple[Artifact, float, str]]:
        query = self._normalized_features(query_features)
        if query is None:
            return []

        scored = []
        for artifact in self.artifacts:
            if artifact.id == exclude_artifact_id or artifact.features is None:
                continue
            candidate = self._normalized_features(artifact.features)
            if candidate is None:
                continue
            score = float(torch.dot(query, candidate).item())
            scored.append((artifact, score, f"creator_{artifact.creator_id}"))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _lineage_path_from_metadata(self, metadata: Optional[Dict[str, object]]) -> List[int]:
        if not metadata:
            return []
        return [int(pid) for pid in metadata.get('domain_lineage_path', []) if pid is not None]

    def _lineage_root_from_metadata(self, metadata: Optional[Dict[str, object]]) -> Optional[int]:
        path = self._lineage_path_from_metadata(metadata)
        if path:
            return path[0]
        root = metadata.get('domain_lineage_root_id') if metadata else None
        return None if root is None else int(root)

    def _scored_lineage_candidates(
        self,
        query_artifact_id: Optional[int],
        query_metadata: Optional[Dict[str, object]],
        exclude_artifact_id: Optional[int] = None,
    ) -> List[Tuple[Artifact, float, str]]:
        query = self.get(query_artifact_id) if query_artifact_id is not None else None
        query_metadata = query.metadata if query is not None else (query_metadata or {})

        query_path = self._lineage_path_from_metadata(query_metadata)
        query_root = self._lineage_root_from_metadata(query_metadata)
        query_parent_id = query_metadata.get('domain_parent_id')
        if query_parent_id is not None:
            query_parent_id = int(query_parent_id)

        scored: List[Tuple[Artifact, float, str]] = []
        for candidate in self.artifacts:
            if candidate.id == exclude_artifact_id:
                continue

            cand_metadata = candidate.metadata
            cand_path = self._lineage_path_from_metadata(cand_metadata)
            cand_parent_id = cand_metadata.get('domain_parent_id')
            if cand_parent_id is not None:
                cand_parent_id = int(cand_parent_id)
            cand_root = self._lineage_root_from_metadata(cand_metadata)

            score = None
            relation_type = None

            if query_parent_id is not None and candidate.id == query_parent_id:
                score, relation_type = 1.0, 'parent'
            elif query_artifact_id is not None and cand_parent_id == query_artifact_id:
                score, relation_type = 0.95, 'child'
            elif query_path and candidate.id in query_path[:-1]:
                idx = query_path.index(candidate.id)
                depth = len(query_path) - 1 - idx
                score, relation_type = max(0.2, 0.9 - 0.1 * (depth - 1)), 'ancestor'
            elif query_artifact_id is not None and cand_path and query_artifact_id in cand_path[:-1]:
                idx = cand_path.index(query_artifact_id)
                depth = len(cand_path) - 1 - idx
                score, relation_type = max(0.2, 0.85 - 0.1 * (depth - 1)), 'descendant'
            elif query_parent_id is not None and cand_parent_id == query_parent_id:
                score, relation_type = 0.8, 'sibling'
            elif query_root is not None and cand_root == query_root:
                score, relation_type = 0.55, 'same_lineage'

            if score is not None:
                scored.append((candidate, float(score), relation_type))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _strategy_target(self, strategy: str, strategy_value: Optional[float]) -> float:
        if strategy_value is not None:
            return min(1.0, max(0.0, float(strategy_value)))
        return {'nearest': 0.0, 'mid': 0.5, 'far': 1.0}[strategy]

    def _bucket_label(self, position: float) -> str:
        if position <= (1.0 / 3.0):
            return 'close'
        if position <= (2.0 / 3.0):
            return 'moderate'
        return 'far'

    def _select_scored_candidate(
        self,
        scored: List[Tuple[Artifact, float, str]],
        retrieval_mode: str,
        fallback_exclude_artifact_id: Optional[int],
        strategy: str,
        strategy_value: Optional[float],
        preferred_novelty: Optional[float],
        current_novelty: Optional[float],
    ) -> Tuple[Optional[Artifact], Dict[str, object]]:
        if not scored:
            artifact, info = self.random_artifact(exclude_artifact_id=fallback_exclude_artifact_id)
            info['retrieval_mode'] = retrieval_mode
            if info.get('relation_type') == 'random':
                info['relation_type'] = f'{retrieval_mode}_fallback_random'
            info['strategy'] = strategy
            info['strategy_value'] = strategy_value
            return artifact, info

        n = len(scored)
        target = self._strategy_target(strategy, strategy_value)
        window = 0.18 if strategy in {'nearest', 'far'} else 0.22

        candidate_indices = []
        for idx in range(n):
            position = idx / max(1, n - 1)
            if abs(position - target) <= window:
                candidate_indices.append(idx)
        if not candidate_indices:
            nearest_idx = min(range(n), key=lambda idx: abs((idx / max(1, n - 1)) - target))
            candidate_indices = [nearest_idx]

        preferred = 0.5 if preferred_novelty is None else float(preferred_novelty)
        baseline_gap = None if current_novelty is None else abs(float(current_novelty) - preferred)

        def objective(idx: int):
            position = idx / max(1, n - 1)
            predicted_gap = abs(position - preferred)
            improvement = 0.0 if baseline_gap is None else (baseline_gap - predicted_gap)
            raw_score = scored[idx][1]
            return (predicted_gap, abs(position - target), -improvement, -raw_score)

        selected_idx = min(candidate_indices, key=objective)
        artifact, score, relation_type = scored[selected_idx]
        position = selected_idx / max(1, n - 1)
        baseline_gap = None if current_novelty is None else abs(float(current_novelty) - preferred)
        predicted_gap = abs(position - preferred)
        motivation_improvement = None if baseline_gap is None else float(baseline_gap - predicted_gap)

        return artifact, {
            'retrieval_mode': retrieval_mode,
            'relation_type': relation_type,
            'score': float(score),
            'strategy': strategy,
            'strategy_value': float(target),
            'rank': int(selected_idx),
            'pool_size': int(n),
            'bucket': self._bucket_label(position),
            'estimated_novelty': float(position),
            'motivation_gap': float(predicted_gap),
            'motivation_improvement': motivation_improvement,
        }
