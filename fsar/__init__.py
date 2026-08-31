"""Paper-faithful Task-Adapter++ few-shot action-recognition utilities."""

from .corpus_v2 import CorpusV2, StageCandidate
from .fusion import fuse_scores
from .model import EpisodicTaskAdapter, TaskAdapter
from .ot import OrderAwareOptimalTransport

__all__ = [
    "CorpusV2",
    "EpisodicTaskAdapter",
    "OrderAwareOptimalTransport",
    "StageCandidate",
    "TaskAdapter",
    "fuse_scores",
]
