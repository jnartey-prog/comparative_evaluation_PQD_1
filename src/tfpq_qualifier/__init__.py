"""Manifest-driven power-quality descriptor qualification software."""

from .models import Representation, SignalRecord
from .representation_features import extract_representation_features
from .synthetic import (
    ManifestCondition,
    SyntheticDataGenerator,
    SyntheticRecord,
    load_conditions,
    record_metadata,
    save_condition_batch,
    save_record,
)
from .transforms import transform

__version__ = "1.0.0"
__all__ = [
    "ManifestCondition",
    "Representation",
    "SignalRecord",
    "SyntheticDataGenerator",
    "SyntheticRecord",
    "extract_representation_features",
    "load_conditions",
    "record_metadata",
    "save_condition_batch",
    "save_record",
    "transform",
]
