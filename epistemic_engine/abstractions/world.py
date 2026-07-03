from __future__ import annotations

import random
from dataclasses import dataclass

from epistemic_engine.abstractions.models import ToyObject


SHAPES = ("round", "angular", "flat")
COLORS = ("red", "blue", "green")
SIZES = ("small", "large")
TEXTURES = ("smooth", "striped")
MATERIALS = ("wood", "metal")


@dataclass(frozen=True)
class ToyWorldConfig:
    seed: int = 31
    train_objects: int = 180
    transfer_objects: int = 60


class HiddenRegularityWorld:
    """Tiny deterministic object world with hidden compositional regularities."""

    def __init__(self, config: ToyWorldConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

    def generate_train(self) -> list[ToyObject]:
        return [self._sample_object(f"train_{index}") for index in range(self.config.train_objects)]

    def generate_transfer(self) -> list[ToyObject]:
        objects: list[ToyObject] = []
        for index in range(self.config.transfer_objects):
            features = {
                "shape": self.rng.choice(SHAPES),
                "color": self.rng.choice(COLORS),
                "size": self.rng.choice(SIZES),
                "texture": self.rng.choice(TEXTURES),
                "material": self.rng.choice(MATERIALS),
            }
            # Skew transfer toward compositions without changing the rules.
            if index % 3 == 0:
                features["shape"] = "round"
                features["color"] = "red"
            if index % 5 == 0:
                features["size"] = "small"
                features["texture"] = "striped"
            objects.append(ToyObject(f"transfer_{index}", features, self._effects(features)))
        return objects

    def _sample_object(self, object_id: str) -> ToyObject:
        features = {
            "shape": self.rng.choice(SHAPES),
            "color": self.rng.choice(COLORS),
            "size": self.rng.choice(SIZES),
            "texture": self.rng.choice(TEXTURES),
            "material": self.rng.choice(MATERIALS),
        }
        return ToyObject(object_id, features, self._effects(features))

    def _effects(self, features: dict[str, str]) -> frozenset[str]:
        effects: set[str] = set()

        if features["shape"] == "round":
            effects.add("rolls")
        if features["size"] == "large":
            effects.add("leaves_trace")
        if features["color"] == "red":
            effects.add("heats")
        if features["material"] == "metal":
            effects.add("conducts")
        if features["texture"] == "striped":
            effects.add("camouflages")

        if features["shape"] == "round" and features["color"] == "red":
            effects.add("glows")
        if features["size"] == "small" and features["texture"] == "striped":
            effects.add("hides")
        if features["material"] == "metal" and features["color"] == "blue":
            effects.add("magnetizes")
        if features["shape"] == "flat" and features["material"] == "wood":
            effects.add("floats")

        return frozenset(effects)
