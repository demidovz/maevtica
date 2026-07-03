# Counterexample Catalogue

## Biased feedback learner
- Failure mode: false_convergence
- Missing assumption: feedback_truth_correlation
- Why it satisfies the five axioms: repeated_interaction, retention, error_feedback, scope_control, nonzero_cost
- Why it still fails: The system receives error feedback, but it is systematically misleading, so retained updates accumulate falsehood.

## Local-trap evolutionary population
- Failure mode: local_trap
- Missing assumption: adequate_exploration
- Why it satisfies the five axioms: repeated_interaction, retention, error_feedback, scope_control, nonzero_cost
- Why it still fails: The population improves locally but cannot reach better regions because variation never crosses the valley.

## Catastrophic forgetting learner
- Failure mode: knowledge_collapse
- Missing assumption: non_destructive_retention
- Why it satisfies the five axioms: repeated_interaction, retention, error_feedback, scope_control, nonzero_cost
- Why it still fails: Retention exists, but updates overwrite prior competencies faster than they consolidate.

## Oscillating research collective
- Failure mode: permanent_oscillation
- Missing assumption: stable_update_dynamics
- Why it satisfies the five axioms: repeated_interaction, retention, error_feedback, scope_control, nonzero_cost
- Why it still fails: The collective reacts to delayed feedback with overcorrection, cycling between incompatible policies.

## Fragmented alien civilization
- Failure mode: communication_failure
- Missing assumption: integration_channel
- Why it satisfies the five axioms: repeated_interaction, retention, error_feedback, scope_control, nonzero_cost
- Why it still fails: Subgroups learn locally, but no integration channel lets local knowledge become civilization-level knowledge.

## Adversarial game-theoretic agents
- Failure mode: Goodharted_feedback
- Missing assumption: incentive_compatibility
- Why it satisfies the five axioms: repeated_interaction, retention, error_feedback, scope_control, nonzero_cost
- Why it still fails: Agents satisfy the axioms but manipulate feedback, so retained knowledge tracks payoffs rather than truth.

## Self-modifying agent
- Failure mode: identity_drift
- Missing assumption: goal_preservation
- Why it satisfies the five axioms: repeated_interaction, retention, error_feedback, scope_control, nonzero_cost
- Why it still fails: The system accumulates knowledge only if self-modification preserves the criterion of improvement.
