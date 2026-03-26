"""Simple baseline policies."""
from epistemic_engine.policies.hybrid_memory import HybridMemoryPolicy
from epistemic_engine.policies.mode_memory import ModeMemoryPolicy
from epistemic_engine.policies.memory import MemoryAugmentedInformationGainPolicy
from epistemic_engine.policies.question_type_memory import QuestionTypeMemoryPolicy
from epistemic_engine.policies.switch_memory import (
    AdaptiveShiftMemoryPolicy,
    ConfirmedSwitchMemoryPolicy,
    LatentAdaptiveShiftMemoryPolicy,
    PersistentShiftMemoryPolicy,
    ReactivatingSwitchMemoryPolicy,
    SwitchAwareHybridMemoryPolicy,
)

__all__ = [
    "HybridMemoryPolicy",
    "MemoryAugmentedInformationGainPolicy",
    "ModeMemoryPolicy",
    "AdaptiveShiftMemoryPolicy",
    "ConfirmedSwitchMemoryPolicy",
    "LatentAdaptiveShiftMemoryPolicy",
    "PersistentShiftMemoryPolicy",
    "QuestionTypeMemoryPolicy",
    "ReactivatingSwitchMemoryPolicy",
    "SwitchAwareHybridMemoryPolicy",
]
