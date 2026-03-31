from __future__ import annotations

import math
from dataclasses import dataclass

from epistemic_engine.beliefs.state import normalize


@dataclass(frozen=True)
class ShiftLatentState:
    mode_support: dict[str, float]
    group_support: dict[str, float]
    top_mode: str
    top_group: str
    default_second_mode: str
    candidate_mode: str
    profile_candidate_mode: str
    hypothesis_candidate_mode: str
    surprise_pressure: float
    ambiguity_pressure: float
    base_switch_pressure: float
    anomaly_pressure: float
    persistence_pressure: float
    rebound_pressure: float
    profile_shift_risk: float
    hypothesis_switch_risk: float
    false_alarm_risk: float
    persistent_shift_risk: float
    aggressive_gate: float
    aggressive_switch_pressure: float
    cautious_switch_pressure: float
    switch_pressure: float


@dataclass(frozen=True)
class ShiftLatentUpdater:
    recent_window: int
    surprise_floor: float
    surprise_scale: float
    anomaly_scale: float
    persistence_scale: float
    rebound_penalty: float
    streak_bonus: float
    rebound_scale: float
    false_alarm_scale: float
    neutral_gate: float
    min_aggressive_gate: float
    same_group_margin: float

    def infer(self, *, state, environment, mode_memory) -> ShiftLatentState:
        return infer_shift_latent(
            state=state,
            environment=environment,
            mode_memory=mode_memory,
            recent_window=self.recent_window,
            surprise_floor=self.surprise_floor,
            surprise_scale=self.surprise_scale,
            anomaly_scale=self.anomaly_scale,
            persistence_scale=self.persistence_scale,
            rebound_penalty=self.rebound_penalty,
            streak_bonus=self.streak_bonus,
            rebound_scale=self.rebound_scale,
            false_alarm_scale=self.false_alarm_scale,
            neutral_gate=self.neutral_gate,
            min_aggressive_gate=self.min_aggressive_gate,
            same_group_margin=self.same_group_margin,
        )


def infer_shift_latent(
    *,
    state,
    environment,
    mode_memory,
    recent_window: int,
    surprise_floor: float,
    surprise_scale: float,
    anomaly_scale: float,
    persistence_scale: float,
    rebound_penalty: float,
    streak_bonus: float,
    rebound_scale: float,
    false_alarm_scale: float,
    neutral_gate: float,
    min_aggressive_gate: float,
    same_group_margin: float,
) -> ShiftLatentState:
    memory_mode_support = mode_memory.support(state.history, environment.mode_ids())
    belief_mode_support = belief_conditioned_mode_support(state=state, environment=environment)
    mode_support = normalize(
        {
            mode_id: 0.45 * memory_mode_support.get(mode_id, 0.0)
            + 0.55 * belief_mode_support.get(mode_id, 0.0)
            for mode_id in environment.mode_ids()
        }
    )
    ordered_modes = sorted(
        mode_support.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    group_support = aggregate_group_support(mode_support=mode_support, environment=environment)
    top_mode, top_probability = ordered_modes[0]
    default_second_mode, second_probability = (
        ordered_modes[1]
        if len(ordered_modes) > 1
        else (top_mode, 0.0)
    )
    top_group = mode_group_id(environment=environment, mode_id=top_mode)

    if not state.history:
        return ShiftLatentState(
            mode_support=mode_support,
            group_support=group_support,
            top_mode=top_mode,
            top_group=top_group,
            default_second_mode=default_second_mode,
            candidate_mode=default_second_mode,
            profile_candidate_mode=top_mode,
            hypothesis_candidate_mode=default_second_mode,
            surprise_pressure=0.0,
            ambiguity_pressure=0.0,
            base_switch_pressure=0.0,
            anomaly_pressure=0.0,
            persistence_pressure=0.0,
            rebound_pressure=0.0,
            profile_shift_risk=0.0,
            hypothesis_switch_risk=0.0,
            false_alarm_risk=0.0,
            persistent_shift_risk=0.0,
            aggressive_gate=neutral_gate,
            aggressive_switch_pressure=0.0,
            cautious_switch_pressure=0.0,
            switch_pressure=0.0,
        )

    mean_surprise = 0.0
    for observation in state.history:
        likelihood = environment.mode_likelihood(
            observation.action_id,
            observation.outcome,
            top_mode,
        )
        mean_surprise += -math.log(max(likelihood, 1e-6))
    mean_surprise /= len(state.history)

    surprise_pressure = min(
        1.0,
        max(0.0, (mean_surprise - surprise_floor) / max(surprise_scale, 1e-9)),
    )
    mode_gap = top_probability - second_probability
    ambiguity_pressure = min(1.0, max(0.0, 1.0 - mode_gap))
    base_switch_pressure = min(
        1.0,
        max(0.0, 0.6 * surprise_pressure + 0.4 * ambiguity_pressure),
    )

    profile_candidate_mode = best_recent_alternative_mode(
        state=state,
        environment=environment,
        top_mode=top_mode,
        default_second_mode=default_second_mode,
        recent_window=recent_window,
        same_group_margin=same_group_margin,
        require_same_group=True,
    )
    hypothesis_candidate_mode = best_recent_alternative_mode(
        state=state,
        environment=environment,
        top_mode=top_mode,
        default_second_mode=default_second_mode,
        recent_window=recent_window,
        same_group_margin=same_group_margin,
        require_same_group=False,
    )
    (
        profile_anomaly_pressure,
        profile_persistence_pressure,
        profile_rebound_pressure,
    ) = recent_shift_signals(
        state=state,
        environment=environment,
        top_mode=top_mode,
        candidate_mode=profile_candidate_mode,
        recent_window=recent_window,
        anomaly_scale=anomaly_scale,
        persistence_scale=persistence_scale,
        rebound_penalty=rebound_penalty,
        streak_bonus=streak_bonus,
        rebound_scale=rebound_scale,
    )
    (
        hypothesis_anomaly_pressure,
        hypothesis_persistence_pressure,
        hypothesis_rebound_pressure,
    ) = recent_shift_signals(
        state=state,
        environment=environment,
        top_mode=top_mode,
        candidate_mode=hypothesis_candidate_mode,
        recent_window=recent_window,
        anomaly_scale=anomaly_scale,
        persistence_scale=persistence_scale,
        rebound_penalty=rebound_penalty,
        streak_bonus=streak_bonus,
        rebound_scale=rebound_scale,
    )
    anomaly_pressure = max(profile_anomaly_pressure, hypothesis_anomaly_pressure)
    persistence_pressure = max(
        profile_persistence_pressure,
        hypothesis_persistence_pressure,
    )
    rebound_pressure = max(profile_rebound_pressure, hypothesis_rebound_pressure)

    profile_shift_evidence = min(
        1.0,
        max(
            0.0,
            0.25 * profile_anomaly_pressure
            + 0.75 * profile_persistence_pressure
            - 0.25 * profile_rebound_pressure,
        ),
    )
    profile_shift_risk = min(
        1.0,
        max(0.0, base_switch_pressure * profile_shift_evidence),
    )

    hypothesis_candidate_group = mode_group_id(
        environment=environment,
        mode_id=hypothesis_candidate_mode,
    )
    top_group_support = group_support.get(top_group, 0.0)
    candidate_group_support = (
        group_support.get(hypothesis_candidate_group, 0.0)
        if hypothesis_candidate_group != top_group
        else 0.0
    )
    group_competition = candidate_group_support / max(
        top_group_support + candidate_group_support,
        1e-9,
    )
    hypothesis_switch_evidence = min(
        1.0,
        max(
            0.0,
            0.20 * hypothesis_anomaly_pressure
            + 0.80 * hypothesis_persistence_pressure
            - 0.45 * hypothesis_rebound_pressure,
        ),
    )
    hypothesis_switch_risk = min(
        1.0,
        max(
            0.0,
            base_switch_pressure
            * hypothesis_switch_evidence
            * (0.35 + 0.65 * group_competition),
        ),
    )

    false_alarm_risk = min(
        1.0,
        max(
            0.0,
            0.50 * profile_shift_risk
            + 0.30 * profile_rebound_pressure
            + 0.20 * max(0.0, profile_shift_risk - hypothesis_switch_risk),
        ),
    )
    persistent_shift_risk = min(
        1.0,
        max(
            0.0,
            max(profile_shift_risk, hypothesis_switch_risk),
        ),
    )
    aggressive_gate = aggressive_switch_gate(
        hypothesis_switch_risk=hypothesis_switch_risk,
        false_alarm_risk=false_alarm_risk,
        false_alarm_scale=false_alarm_scale,
        neutral_gate=neutral_gate,
        min_aggressive_gate=min_aggressive_gate,
    )
    aggressive_pressure = base_switch_pressure * hypothesis_switch_risk
    cautious_pressure = base_switch_pressure * persistent_shift_risk
    switch_pressure = (
        aggressive_gate * aggressive_pressure
        + (1.0 - aggressive_gate) * cautious_pressure
    )
    candidate_mode = (
        hypothesis_candidate_mode
        if hypothesis_switch_risk >= profile_shift_risk
        else profile_candidate_mode
    )

    return ShiftLatentState(
        mode_support=mode_support,
        group_support=group_support,
        top_mode=top_mode,
        top_group=top_group,
        default_second_mode=default_second_mode,
        candidate_mode=candidate_mode,
        profile_candidate_mode=profile_candidate_mode,
        hypothesis_candidate_mode=hypothesis_candidate_mode,
        surprise_pressure=surprise_pressure,
        ambiguity_pressure=ambiguity_pressure,
        base_switch_pressure=base_switch_pressure,
        anomaly_pressure=anomaly_pressure,
        persistence_pressure=persistence_pressure,
        rebound_pressure=rebound_pressure,
        profile_shift_risk=profile_shift_risk,
        hypothesis_switch_risk=hypothesis_switch_risk,
        false_alarm_risk=false_alarm_risk,
        persistent_shift_risk=persistent_shift_risk,
        aggressive_gate=aggressive_gate,
        aggressive_switch_pressure=aggressive_pressure,
        cautious_switch_pressure=cautious_pressure,
        switch_pressure=min(1.0, max(0.0, switch_pressure)),
    )


def best_recent_alternative_mode(
    *,
    state,
    environment,
    top_mode: str,
    default_second_mode: str,
    recent_window: int,
    same_group_margin: float,
    require_same_group: bool | None = None,
) -> str:
    if len(environment.mode_ids()) <= 1 or not state.history:
        return default_second_mode

    recent_history = state.history[-recent_window:]
    best_mode = default_second_mode
    best_score = float("-inf")
    best_same_group_mode = default_second_mode
    best_same_group_score = float("-inf")
    top_group = mode_group_id(environment=environment, mode_id=top_mode)

    for mode_id in environment.mode_ids():
        if mode_id == top_mode:
            continue
        candidate_group = mode_group_id(environment=environment, mode_id=mode_id)
        if require_same_group is True and candidate_group != top_group:
            continue
        if require_same_group is False and candidate_group == top_group:
            continue
        score = 0.0
        for index, observation in enumerate(recent_history, start=1):
            weight = index / len(recent_history)
            candidate_likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                mode_id,
            )
            top_likelihood = environment.mode_likelihood(
                observation.action_id,
                observation.outcome,
                top_mode,
            )
            score += weight * (
                math.log(max(candidate_likelihood, 1e-6))
                - math.log(max(top_likelihood, 1e-6))
            )
        if score > best_score:
            best_score = score
            best_mode = mode_id
        if candidate_group == top_group and score > best_same_group_score:
            best_same_group_score = score
            best_same_group_mode = mode_id

    if best_score == float("-inf"):
        return top_mode
    if best_same_group_score >= best_score - same_group_margin:
        return best_same_group_mode
    return best_mode


def belief_conditioned_mode_support(*, state, environment) -> dict[str, float]:
    scores = {}
    for mode_id in environment.mode_ids():
        compatibility = 0.0
        for hypothesis_id, mode_weight in environment.hypotheses_given_mode(mode_id).items():
            compatibility += state.probabilities.get(hypothesis_id, 0.0) * mode_weight
        scores[mode_id] = compatibility + 1e-9
    return normalize(scores)


def recent_shift_signals(
    *,
    state,
    environment,
    top_mode: str,
    candidate_mode: str,
    recent_window: int,
    anomaly_scale: float,
    persistence_scale: float,
    rebound_penalty: float,
    streak_bonus: float,
    rebound_scale: float,
) -> tuple[float, float, float]:
    if candidate_mode == top_mode or not state.history:
        return 0.0, 0.0, 0.0

    recent_history = state.history[-recent_window:]
    weighted_positive = 0.0
    weighted_negative = 0.0
    total_weight = 0.0
    latest_advantage = 0.0
    positive_streak = 0
    prior_positive = 0.0
    prior_positive_weight = 0.0

    for reverse_index, observation in enumerate(reversed(recent_history), start=1):
        forward_index = len(recent_history) - reverse_index + 1
        weight = forward_index / len(recent_history)
        candidate_likelihood = environment.mode_likelihood(
            observation.action_id,
            observation.outcome,
            candidate_mode,
        )
        top_likelihood = environment.mode_likelihood(
            observation.action_id,
            observation.outcome,
            top_mode,
        )
        advantage = (
            math.log(max(candidate_likelihood, 1e-6))
            - math.log(max(top_likelihood, 1e-6))
        )
        if reverse_index == 1:
            latest_advantage = advantage
        if advantage > 0.0 and positive_streak == reverse_index - 1:
            positive_streak += 1
        if reverse_index > 1:
            prior_positive += weight * max(0.0, advantage)
            prior_positive_weight += weight

        weighted_positive += weight * max(0.0, advantage)
        weighted_negative += weight * max(0.0, -advantage)
        total_weight += weight

    weighted_positive /= max(total_weight, 1e-9)
    weighted_negative /= max(total_weight, 1e-9)
    anomaly_pressure = min(
        1.0,
        max(0.0, latest_advantage / max(anomaly_scale, 1e-9)),
    )
    streak_pressure = positive_streak / max(len(recent_history), 1)
    persistence_score = (
        weighted_positive
        - rebound_penalty * weighted_negative
        + streak_bonus * streak_pressure
    )
    persistence_pressure = min(
        1.0,
        max(0.0, persistence_score / max(persistence_scale, 1e-9)),
    )
    prior_positive /= max(prior_positive_weight, 1e-9)
    latest_negative_pressure = min(
        1.0,
        max(0.0, -latest_advantage / max(anomaly_scale, 1e-9)),
    )
    prior_support_pressure = min(
        1.0,
        max(0.0, prior_positive / max(persistence_scale, 1e-9)),
    )
    rebound_pressure = min(
        1.0,
        max(
            0.0,
            latest_negative_pressure * (0.5 + 0.5 * prior_support_pressure)
            / max(rebound_scale, 1e-9),
        ),
    )
    return anomaly_pressure, persistence_pressure, rebound_pressure


def aggressive_switch_gate(
    *,
    hypothesis_switch_risk: float,
    false_alarm_risk: float,
    false_alarm_scale: float,
    neutral_gate: float,
    min_aggressive_gate: float,
) -> float:
    if hypothesis_switch_risk <= 1e-9 and false_alarm_risk <= 1e-9:
        return neutral_gate

    denominator = hypothesis_switch_risk + false_alarm_scale * false_alarm_risk
    if denominator <= 1e-9:
        return min_aggressive_gate

    gate = hypothesis_switch_risk / denominator
    return min(1.0, max(min_aggressive_gate, gate))


def mode_group_id(*, environment, mode_id: str) -> str:
    mode_group = getattr(environment, "mode_group", None)
    if callable(mode_group):
        return mode_group(mode_id)
    return mode_id


def aggregate_group_support(*, mode_support: dict[str, float], environment) -> dict[str, float]:
    scores: dict[str, float] = {}
    for mode_id, support in mode_support.items():
        group_id = mode_group_id(environment=environment, mode_id=mode_id)
        scores[group_id] = scores.get(group_id, 0.0) + support
    return normalize(scores)


def latent_mode_support_to_hypotheses(environment, latent_state: ShiftLatentState) -> dict[str, float]:
    scores = environment.mode_support_to_hypotheses(latent_state.mode_support)
    return normalize(scores)
