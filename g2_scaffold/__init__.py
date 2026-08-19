"""Isolated CPU-testable scaffold for the G2 hierarchical decoder."""

from .decoder import DecodedEvent, Hypothesis, decode_batch, decode_event
from .losses import hierarchical_loss
from .model import G2CandidateScorer
from .targets import targets_to_g2

__all__ = [
    "DecodedEvent",
    "G2CandidateScorer",
    "Hypothesis",
    "decode_batch",
    "decode_event",
    "hierarchical_loss",
    "targets_to_g2",
]
