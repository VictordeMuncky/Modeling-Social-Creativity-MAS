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
    VALID_STRATEGIES = ("nearest", "mid", "far", "learned")
    VALID_SELECTION_POLICIES = ("simple",)
    STRATEGY_WINDOW = 0.05

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
        return artifact_id is not None and int(artifact_id) in self.artifact_ids

    def get(self, artifact_id: Optional[int]) -> Optional[Artifact]:
        if artifact_id is None:
            return None
        return self.artifact_by_id.get(int(artifact_id))

    def prepare_artifact(self, artifact: Artifact, step: int, source: str = "generation"):
        """Populate or refresh artifact metadata before domain entry."""
        metadata = artifact.metadata
        metadata['artifact_id'] = artifact.id
        metadata['producer_id'] = artifact.producer_id
        metadata['parent_ids'] = [pid for pid in (artifact.parent1_id, artifact.parent2_id) if pid is not None]
        metadata['generation_step'] = step
        metadata['domain_source'] = source
        metadata['feature_dims'] = None if artifact.features is None else int(artifact.features.shape[0])

        inherited_domain_parent_id = metadata.get('domain_parent_id')
        metadata['domain_parent_id'] = None if inherited_domain_parent_id is None else int(inherited_domain_parent_id)

        root_creator_id = artifact.creator_id
        parent_depths = []
        for parent_id in metadata['parent_ids']:
            parent = self.get(parent_id)
            if parent is not None:
                root_creator_id = parent.metadata.get('root_creator_id', root_creator_id)
                parent_depths.append(int(parent.metadata.get('lineage_depth', 0)))

        metadata['root_creator_id'] = root_creator_id
        metadata['lineage_depth'] = (max(parent_depths) + 1) if parent_depths else (0 if not metadata['parent_ids'] else 1)
        metadata['lineage_signature'] = f"{metadata['root_creator_id']}:{artifact.parent1_id}:{artifact.parent2_id}"

        self.update_similarity_metadata(artifact)
        return artifact

    def add_artifact(self, artifact: Artifact, accepted_by: int, step: int) -> bool:
        """Register an artifact in the shared domain memory and update indexes."""
        self.prepare_artifact(
            artifact,
            step=artifact.metadata.get('generation_step', step) or step,
            source=artifact.metadata.get('domain_source', 'generation'),
        )

        domain_parent_id = artifact.metadata.get('domain_parent_id')
        artifact.metadata['domain_parent_id'] = None if domain_parent_id is None else int(domain_parent_id)

        is_new = artifact.id not in self.artifact_ids
        if is_new:
            self.artifacts.append(artifact)
            self.artifact_ids.add(artifact.id)
            self.artifact_by_id[artifact.id] = artifact

        self.by_creator[artifact.creator_id].append(artifact.id)
        artifact.add_domain_entry(accepted_by, step)
        self.update_similarity_metadata(artifact)
        return is_new

    def random_artifact(self, exclude_artifact_id: Optional[int] = None) -> Tuple[Optional[Artifact], Dict[str, object]]:
        candidates = [a for a in self.artifacts if a.id != exclude_artifact_id]
        if not candidates:
            return None, {
                'retrieval_mode': 'flat',
                'relation_type': 'empty',
                'score': None,
                'strategy': None,
                'strategy_value': None,
                'selection_policy': None,
                'rank': None,
                'pool_size': 0,
                'bucket': None,
                'estimated_novelty': None,
                'fallback_random': False,
            }

        artifact = random.choice(candidates)
        return artifact, {
            'retrieval_mode': 'flat',
            'relation_type': 'random',
            'score': 1.0,
            'strategy': None,
            'strategy_value': None,
            'selection_policy': None,
            'rank': 0,
            'pool_size': len(candidates),
            'bucket': 'random',
            'estimated_novelty': None,
            'fallback_random': False,
        }

    def retrieve(
        self,
        query_artifact: Optional[Artifact] = None,
        query_features: Optional[torch.Tensor] = None,
        query_artifact_id: Optional[int] = None,
        query_metadata: Optional[Dict[str, object]] = None,
        mode: Optional[str] = None,
        exclude_artifact_id: Optional[int] = None,
        strategy: str = 'nearest',
        strategy_value: Optional[float] = None,
        selection_policy: str = 'simple',
        preferred_novelty: Optional[float] = None,
    ) -> Tuple[Optional[Artifact], Dict[str, object]]:
        """Retrieve one artifact according to the selected domain structure."""
        mode = mode or self.mode

        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported domain mode '{mode}'.")
        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(f"Unsupported domain strategy '{strategy}'. Expected one of {self.VALID_STRATEGIES}.")
        if selection_policy not in self.VALID_SELECTION_POLICIES:
            raise ValueError(
                f"Unsupported selection policy '{selection_policy}'. "
                f"Expected one of {self.VALID_SELECTION_POLICIES}."
            )

        if not self.artifacts:
            return None, {
                'retrieval_mode': mode,
                'relation_type': 'empty',
                'score': None,
                'strategy': strategy,
                'strategy_value': strategy_value,
                'selection_policy': selection_policy,
                'fallback_random': False,
            }

        if query_artifact is not None:
            query_artifact_id = query_artifact.id
            query_features = query_artifact.features if query_features is None else query_features
            query_metadata = query_artifact.metadata if query_metadata is None else query_metadata

        exclude_id = exclude_artifact_id if exclude_artifact_id is not None else query_artifact_id

        if mode == 'flat':
            artifact, info = self.random_artifact(exclude_artifact_id=exclude_id)
            info['strategy'] = strategy
            info['strategy_value'] = strategy_value
            info['selection_policy'] = selection_policy
            return artifact, info

        if mode == 'similarity':
            scored = self._scored_similarity_candidates(
                query_features=query_features,
                exclude_artifact_id=exclude_id,
            )
        else:
            scored = self._scored_lineage_candidates(
                query_artifact_id=query_artifact_id,
                query_metadata=query_metadata,
                exclude_artifact_id=exclude_id,
            )

        return self._select_scored_candidate(
            scored=scored,
            retrieval_mode=mode,
            fallback_exclude_artifact_id=exclude_id,
            strategy=strategy,
            strategy_value=strategy_value,
            selection_policy=selection_policy,
            preferred_novelty=preferred_novelty,
        )
    def query_related(self, artifact: Artifact, k: int = 5, mode: Optional[str] = None) -> List[Tuple[Artifact, float, str]]:
        """Return top related artifacts according to the selected structure."""
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
            scored = self._scored_lineage_candidates(
                query_artifact_id=artifact.id,
                query_metadata=artifact.metadata,
                exclude_artifact_id=exclude_id,
            )
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

    def _ancestor_chain(self, artifact_id: Optional[int]) -> List[int]:
        chain: List[int] = []
        seen = set()
        current_id = None if artifact_id is None else int(artifact_id)
        while current_id is not None and current_id not in seen:
            seen.add(current_id)
            chain.append(current_id)
            artifact = self.get(current_id)
            if artifact is None:
                break
            next_id = artifact.metadata.get('domain_parent_id')
            current_id = None if next_id is None else int(next_id)
        return chain

    def _scored_lineage_candidates(
        self,
        query_artifact_id: Optional[int],
        query_metadata: Optional[Dict[str, object]],
        exclude_artifact_id: Optional[int] = None,
    ) -> List[Tuple[Artifact, float, str]]:
        if query_artifact_id is not None and self.contains(query_artifact_id):
            query_parent_id = int(query_artifact_id)
            query_ancestor_chain = self._ancestor_chain(query_artifact_id)
        else:
            query_parent_id = None if not query_metadata else query_metadata.get('domain_parent_id')
            query_parent_id = None if query_parent_id is None else int(query_parent_id)
            query_ancestor_chain = self._ancestor_chain(query_parent_id)

        scored: List[Tuple[Artifact, float, str]] = []
        for candidate in self.artifacts:
            if candidate.id == exclude_artifact_id:
                continue

            cand_parent_id = candidate.metadata.get('domain_parent_id')
            cand_parent_id = None if cand_parent_id is None else int(cand_parent_id)
            cand_chain = self._ancestor_chain(candidate.id)

            score = None
            relation_type = None

            if query_parent_id is not None and candidate.id == query_parent_id:
                score, relation_type = 1.0, 'parent'
            elif query_artifact_id is not None and cand_parent_id == query_artifact_id:
                score, relation_type = 0.95, 'child'
            elif query_parent_id is not None and cand_parent_id == query_parent_id:
                score, relation_type = 0.8, 'sibling'
            elif candidate.id in query_ancestor_chain[1:]:
                depth = query_ancestor_chain.index(candidate.id)
                score, relation_type = max(0.2, 0.9 - 0.1 * (depth - 1)), 'ancestor'
            elif query_artifact_id is not None and query_artifact_id in cand_chain[1:]:
                depth = cand_chain.index(query_artifact_id)
                score, relation_type = max(0.2, 0.85 - 0.1 * (depth - 1)), 'descendant'
            elif query_ancestor_chain and cand_chain and query_ancestor_chain[-1] == cand_chain[-1]:
                score, relation_type = 0.55, 'same_lineage'

            if score is not None:
                scored.append((candidate, float(score), relation_type))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _strategy_target(self, strategy: str, strategy_value: Optional[float]) -> float:
        if strategy_value is not None:
            return min(1.0, max(0.0, float(strategy_value)))
        return {'nearest': 0.0, 'mid': 0.5, 'far': 1.0, 'learned': 0.5}[strategy]

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
        selection_policy: str,
        preferred_novelty: Optional[float],
    ) -> Tuple[Optional[Artifact], Dict[str, object]]:
        if not scored:
            artifact, info = self.random_artifact(exclude_artifact_id=fallback_exclude_artifact_id)
            info['retrieval_mode'] = retrieval_mode
            if info.get('relation_type') == 'random':
                info['relation_type'] = f'{retrieval_mode}_fallback_random'
            info['strategy'] = strategy
            info['strategy_value'] = strategy_value
            info['selection_policy'] = selection_policy
            info['fallback_random'] = True
            return artifact, info

        n = len(scored)
        target = self._strategy_target(strategy, strategy_value)
        window = self.STRATEGY_WINDOW

        candidate_indices = []
        for idx in range(n):
            position = idx / max(1, n - 1)
            if abs(position - target) <= window:
                candidate_indices.append(idx)

        if not candidate_indices:
            nearest_idx = min(range(n), key=lambda idx: abs((idx / max(1, n - 1)) - target))
            candidate_indices = [nearest_idx]

        if selection_policy == 'simple':
            selected_idx = random.choice(candidate_indices)
        else:
            raise ValueError(f"Unsupported selection policy '{selection_policy}'.")

        artifact, score, relation_type = scored[selected_idx]
        position = selected_idx / max(1, n - 1)

        return artifact, {
            'retrieval_mode': retrieval_mode,
            'relation_type': relation_type,
            'score': float(score),
            'strategy': strategy,
            'strategy_value': float(target),
            'selection_policy': selection_policy,
            'rank': int(selected_idx),
            'pool_size': int(n),
            'bucket': self._bucket_label(position),
            'estimated_novelty': float(position),
            'fallback_random': False,
        }

