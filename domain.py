"""
Domain Memory Abstraction
=========================

A single shared cultural-memory object for the simulation.

The Domain stores artifacts and metadata once, then supports multiple
organizational interpretations through different retrieval modes:
    - flat       : unstructured archive with random retrieval
    - similarity : feature-space neighborhood retrieval
    - lineage    : ancestry/derivation retrieval
    - popularity : retrieval guided by social-circulation metadata

The goal is to keep the rest of the simulation stable while making the
Domain (explicitly) an experimental variable.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from framework import Artifact


class Domain:
    """
    Single cultural-memory object with multiple retrieval structures.

    Artifacts are stored once. Different domain modes change how the same
    memory is indexed and retrieved.
    """

    VALID_MODES = ("flat", "similarity", "lineage", "popularity")

    def __init__(self, mode: str = "flat"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported domain mode '{mode}'. Expected one of {self.VALID_MODES}.")

        self.mode = mode
        self.artifacts: List[Artifact] = []
        self.artifact_ids = set()
        self.artifact_by_id: Dict[int, Artifact] = {}
        self.by_creator = defaultdict(list)
        self.children_by_parent = defaultdict(list)
        self.popularity_ranking: Dict[int, float] = {}

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

        artifact.refresh_popularity_score()
        self.update_similarity_metadata(artifact)
        return artifact

    def add_artifact(self, artifact: Artifact, accepted_by: int, step: int) -> Tuple[bool, float]:
        """
        Register an artifact in the shared domain memory and update indexes.

        Returns:
            (is_new_artifact, popularity_score)
        """
        self.prepare_artifact(artifact, step=artifact.metadata.get('generation_step', step) or step,
                              source=artifact.metadata.get('domain_source', 'generation'))

        is_new = artifact.id not in self.artifact_ids
        if is_new:
            self.artifacts.append(artifact)
            self.artifact_ids.add(artifact.id)
            self.artifact_by_id[artifact.id] = artifact

        self.by_creator[artifact.creator_id].append(artifact.id)
        for parent_id in artifact.metadata.get('parent_ids', []):
            self.children_by_parent[parent_id].append(artifact.id)

        artifact.add_domain_entry(accepted_by, step)
        artifact.add_like(accepted_by)
        popularity = artifact.refresh_popularity_score()
        self.popularity_ranking[artifact.id] = popularity
        self.update_similarity_metadata(artifact)
        return is_new, popularity

    def random_artifact(self, exclude_artifact_id: Optional[int] = None) -> Tuple[Optional[Artifact], Dict[str, object]]:
        candidates = [a for a in self.artifacts if a.id != exclude_artifact_id]
        if not candidates:
            return None, {'retrieval_mode': 'flat', 'relation_type': 'empty', 'score': None}
        artifact = random.choice(candidates)
        return artifact, {'retrieval_mode': 'flat', 'relation_type': 'random', 'score': 1.0}

    def retrieve(self,
                 query_artifact: Optional[Artifact] = None,
                 query_features: Optional[torch.Tensor] = None,
                 query_artifact_id: Optional[int] = None,
                 mode: Optional[str] = None,
                 exclude_artifact_id: Optional[int] = None) -> Tuple[Optional[Artifact], Dict[str, object]]:
        """
        Retrieve one artifact according to the selected domain structure.
        """
        mode = mode or self.mode
        if not self.artifacts:
            return None, {'retrieval_mode': mode, 'relation_type': 'empty', 'score': None}

        if query_artifact is not None:
            query_artifact_id = query_artifact.id
            query_features = query_artifact.features if query_features is None else query_features

        if mode == 'flat':
            return self.random_artifact(exclude_artifact_id=exclude_artifact_id or query_artifact_id)
        if mode == 'similarity':
            return self._retrieve_similarity(query_features=query_features, exclude_artifact_id=exclude_artifact_id or query_artifact_id)
        if mode == 'lineage':
            return self._retrieve_lineage(query_artifact_id=query_artifact_id, exclude_artifact_id=exclude_artifact_id)
        if mode == 'popularity':
            return self._retrieve_popularity(query_artifact=query_artifact, exclude_artifact_id=exclude_artifact_id or query_artifact_id)
        raise ValueError(f"Unsupported domain mode '{mode}'.")

    def query_related(self,
                      artifact: Artifact,
                      k: int = 5,
                      mode: Optional[str] = None) -> List[Tuple[Artifact, float, str]]:
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
            scored = []
            query = self._normalized_features(artifact.features)
            if query is None:
                return []
            for candidate in candidates:
                cand = self._normalized_features(candidate.features)
                if cand is None:
                    continue
                score = float(torch.dot(query, cand).item())
                scored.append((candidate, score, 'feature_neighbor'))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]

        if mode == 'lineage':
            lineage_scored = []
            related_ids = self._lineage_candidate_scores(artifact.id)
            for candidate_id, (score, relation_type) in related_ids.items():
                candidate = self.get(candidate_id)
                if candidate is not None:
                    lineage_scored.append((candidate, score, relation_type))
            lineage_scored.sort(key=lambda x: x[1], reverse=True)
            return lineage_scored[:k]

        if mode == 'popularity':
            scored = []
            for candidate in candidates:
                score, relation_type = self._popularity_relation_score(artifact, candidate)
                scored.append((candidate, score, relation_type))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:k]

        raise ValueError(f"Unsupported domain mode '{mode}'.")

    def update_similarity_metadata(self, artifact: Artifact):
        metadata = artifact.metadata
        metadata['feature_dims'] = None if artifact.features is None else int(artifact.features.shape[0])

        related, info = self._retrieve_similarity(
            query_features=artifact.features,
            exclude_artifact_id=artifact.id,
        )
        if related is None:
            metadata['nearest_domain_artifact_id'] = None
            metadata['nearest_domain_similarity'] = None
            metadata['domain_cluster_hint'] = None
            return

        metadata['nearest_domain_artifact_id'] = related.id
        metadata['nearest_domain_similarity'] = info.get('score')
        metadata['domain_cluster_hint'] = info.get('relation_type') or f"creator_{related.creator_id}"

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

    def _retrieve_similarity(self,
                             query_features: Optional[torch.Tensor],
                             exclude_artifact_id: Optional[int] = None) -> Tuple[Optional[Artifact], Dict[str, object]]:
        query = self._normalized_features(query_features)
        if query is None:
            return self.random_artifact(exclude_artifact_id=exclude_artifact_id)

        best_artifact = None
        best_score = None
        for artifact in self.artifacts:
            if artifact.id == exclude_artifact_id or artifact.features is None:
                continue
            candidate = self._normalized_features(artifact.features)
            if candidate is None:
                continue
            score = float(torch.dot(query, candidate).item())
            if best_score is None or score > best_score:
                best_score = score
                best_artifact = artifact

        if best_artifact is None:
            return self.random_artifact(exclude_artifact_id=exclude_artifact_id)

        return best_artifact, {
            'retrieval_mode': 'similarity',
            'relation_type': f"creator_{best_artifact.creator_id}",
            'score': best_score,
        }

    def _lineage_ancestors(self, artifact_id: int) -> Dict[int, int]:
        ancestors = {}
        queue = [(artifact_id, 0)]
        visited = {artifact_id}
        while queue:
            current_id, depth = queue.pop(0)
            current = self.get(current_id)
            if current is None:
                continue
            for parent_id in current.metadata.get('parent_ids', []):
                if parent_id in visited:
                    continue
                visited.add(parent_id)
                ancestors[parent_id] = depth + 1
                queue.append((parent_id, depth + 1))
        return ancestors

    def _lineage_descendants(self, artifact_id: int) -> Dict[int, int]:
        descendants = {}
        queue = [(artifact_id, 0)]
        visited = {artifact_id}
        while queue:
            current_id, depth = queue.pop(0)
            child_ids = self.children_by_parent.get(current_id, [])
            for child_id in child_ids:
                if child_id in visited:
                    continue
                visited.add(child_id)
                descendants[child_id] = depth + 1
                queue.append((child_id, depth + 1))
        return descendants

    def _lineage_candidate_scores(self, artifact_id: Optional[int]) -> Dict[int, Tuple[float, str]]:
        if artifact_id is None:
            return {}
        query = self.get(artifact_id)
        if query is None:
            return {}

        scores: Dict[int, Tuple[float, str]] = {}

        for parent_id in query.metadata.get('parent_ids', []):
            if self.contains(parent_id):
                scores[parent_id] = (1.0, 'parent')

        for child_id in self.children_by_parent.get(artifact_id, []):
            if self.contains(child_id):
                prev = scores.get(child_id, (-1.0, 'child'))
                scores[child_id] = max(prev, (0.95, 'child'), key=lambda x: x[0])

        ancestors = self._lineage_ancestors(artifact_id)
        for ancestor_id, depth in ancestors.items():
            score = max(0.2, 0.9 - 0.1 * (depth - 1))
            prev = scores.get(ancestor_id, (-1.0, 'ancestor'))
            scores[ancestor_id] = max(prev, (score, 'ancestor'), key=lambda x: x[0])

        descendants = self._lineage_descendants(artifact_id)
        for descendant_id, depth in descendants.items():
            score = max(0.2, 0.85 - 0.1 * (depth - 1))
            prev = scores.get(descendant_id, (-1.0, 'descendant'))
            scores[descendant_id] = max(prev, (score, 'descendant'), key=lambda x: x[0])

        sibling_ids = set()
        for parent_id in query.metadata.get('parent_ids', []):
            sibling_ids.update(self.children_by_parent.get(parent_id, []))
        sibling_ids.discard(artifact_id)
        for sibling_id in sibling_ids:
            prev = scores.get(sibling_id, (-1.0, 'sibling'))
            scores[sibling_id] = max(prev, (0.8, 'sibling'), key=lambda x: x[0])

        return scores

    def _retrieve_lineage(self,
                          query_artifact_id: Optional[int],
                          exclude_artifact_id: Optional[int] = None) -> Tuple[Optional[Artifact], Dict[str, object]]:
        scores = self._lineage_candidate_scores(query_artifact_id)
        candidates = []
        weights = []
        relations = []
        for candidate_id, (score, relation_type) in scores.items():
            if candidate_id == exclude_artifact_id:
                continue
            artifact = self.get(candidate_id)
            if artifact is None:
                continue
            candidates.append(artifact)
            weights.append(max(score, 1e-6))
            relations.append((score, relation_type))

        if not candidates:
            artifact, info = self.random_artifact(exclude_artifact_id=exclude_artifact_id or query_artifact_id)
            info['retrieval_mode'] = 'lineage'
            if info.get('relation_type') == 'random':
                info['relation_type'] = 'lineage_fallback_random'
            return artifact, info

        idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        score, relation_type = relations[idx]
        return candidates[idx], {
            'retrieval_mode': 'lineage',
            'relation_type': relation_type,
            'score': float(score),
        }

    def _audience_overlap(self, left: Artifact, right: Artifact) -> float:
        left_audience = set(left.metadata.get('viewers', [])) | set(left.metadata.get('liked_by', [])) | set(left.metadata.get('accepted_by', []))
        right_audience = set(right.metadata.get('viewers', [])) | set(right.metadata.get('liked_by', [])) | set(right.metadata.get('accepted_by', []))
        if not left_audience or not right_audience:
            return 0.0
        union = left_audience | right_audience
        if not union:
            return 0.0
        return len(left_audience & right_audience) / len(union)

    def _popularity_relation_score(self, query_artifact: Optional[Artifact], candidate: Artifact) -> Tuple[float, str]:
        candidate_popularity = float(candidate.metadata.get('popularity_score', candidate.refresh_popularity_score()))
        if query_artifact is None:
            return max(1.0, candidate_popularity + 1.0), 'popular'

        query_popularity = float(query_artifact.metadata.get('popularity_score', query_artifact.refresh_popularity_score()))
        overlap = self._audience_overlap(query_artifact, candidate)
        closeness = 1.0 / (1.0 + abs(candidate_popularity - query_popularity))
        domain_reinforcement = 1.0 + float(candidate.metadata.get('domain_entry_count', 0))
        score = (2.0 * overlap) + (1.5 * closeness) + (0.2 * domain_reinforcement) + (0.1 * candidate_popularity)
        relation_type = 'audience_overlap' if overlap > 0 else 'popularity_band'
        return score, relation_type

    def _retrieve_popularity(self,
                             query_artifact: Optional[Artifact],
                             exclude_artifact_id: Optional[int] = None) -> Tuple[Optional[Artifact], Dict[str, object]]:
        candidates = []
        weights = []
        relations = []
        for artifact in self.artifacts:
            if artifact.id == exclude_artifact_id:
                continue
            score, relation_type = self._popularity_relation_score(query_artifact, artifact)
            score = max(score, 1e-6)
            candidates.append(artifact)
            weights.append(score)
            relations.append((score, relation_type))

        if not candidates:
            return None, {'retrieval_mode': 'popularity', 'relation_type': 'empty', 'score': None}

        idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
        score, relation_type = relations[idx]
        return candidates[idx], {
            'retrieval_mode': 'popularity',
            'relation_type': relation_type,
            'score': float(score),
        }
