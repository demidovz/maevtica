from __future__ import annotations

from epistemic_engine.beliefs.state import normalize, top_hypothesis
from epistemic_engine.models import BeliefState, Observation, RevisionEvent


def refresh_shift_latent(state: BeliefState, environment, policy) -> None:
    updater = getattr(policy, "shift_latent_updater", None)
    mode_memory = getattr(policy, "mode_memory", None)
    if updater is None or mode_memory is None:
        return

    latent_state = updater.infer(
        state=state,
        environment=environment,
        mode_memory=mode_memory,
    )
    state.shift_latent = latent_state
    if len(state.shift_latent_history) <= len(state.history):
        state.shift_latent_history.append(latent_state)
    else:
        state.shift_latent_history[-1] = latent_state


def apply_observation(
    state: BeliefState,
    environment,
    observation: Observation,
    policy=None,
) -> None:
    previous_top, previous_confidence = top_hypothesis(state)
    posterior_scores: dict[str, float] = {}
    observation_weight = 1.0
    if policy is not None:
        weight_fn = getattr(policy, "observation_weight", None)
        if callable(weight_fn):
            observation_weight = min(
                1.0,
                max(
                    0.0,
                    weight_fn(
                        state=state,
                        environment=environment,
                        observation=observation,
                    ),
                ),
            )

    for hypothesis_id, prior in state.probabilities.items():
        likelihood = environment.likelihood(
            observation.action_id,
            observation.outcome,
            hypothesis_id,
        )
        tempered_likelihood = max(likelihood, 1e-9) ** observation_weight
        posterior_scores[hypothesis_id] = prior * tempered_likelihood

    state.probabilities = normalize(posterior_scores)
    state.asked_actions.append(observation.action_id)
    state.total_cost += observation.cost
    state.history.append(observation)

    new_top, new_confidence = top_hypothesis(state)
    if new_top != previous_top:
        state.revisions.append(
            RevisionEvent(
                previous_top=previous_top,
                new_top=new_top,
                reason=(
                    f"Top hypothesis switched after {observation.action_id} -> "
                    f"{observation.outcome}"
                ),
            )
        )
    elif new_confidence > previous_confidence + 0.15:
        state.revisions.append(
            RevisionEvent(
                previous_top=previous_top,
                new_top=new_top,
                reason=(
                    f"Confidence in {new_top} jumped after "
                    f"{observation.action_id} -> {observation.outcome}"
                ),
            )
        )
