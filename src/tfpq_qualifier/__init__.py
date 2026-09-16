"""Simple public interface for TFPQ Qualifier."""

from .artifacts import generate_artifacts
from .evidence import evaluate_feature
from .features import extract_features
from .models import (
    Evidence,
    FeatureRecord,
    HoldoutConfirmation,
    Qualification,
    QualificationPolicy,
    Representation,
    SignalRecord,
    SignalSpec,
    StudyConfig,
    StudyResult,
)
from .pipeline import run
from .qualification import confirm_holdout, qualify
from .representation_features import extract_representation_features
from .signals import build_dataset, generate_signal
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
    "Evidence",
    "FeatureRecord",
    "HoldoutConfirmation",
    "ManifestCondition",
    "Qualification",
    "QualificationPolicy",
    "Representation",
    "SignalRecord",
    "SignalSpec",
    "StudyConfig",
    "StudyResult",
    "SyntheticDataGenerator",
    "SyntheticRecord",
    "build_dataset",
    "confirm_holdout",
    "evaluate_feature",
    "extract_features",
    "extract_representation_features",
    "generate_artifacts",
    "generate_signal",
    "load_conditions",
    "qualify",
    "record_metadata",
    "run",
    "save_condition_batch",
    "save_record",
    "transform",
]
