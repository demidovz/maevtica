# World Catalogue

## w01_baseline_science: Baseline learnable universe
Stable causal world with fallible agents, finite memory, noisy observation, and reusable regularities.

- physics: stable
- logic: classical
- memory_reliability: fallible
- communication: medium_bandwidth
- agent_lifespan: finite
- resource_availability: scarce
- observation_noise: moderate
- computation_cost: moderate
- learning_cost: moderate
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: compositional
- time: linear
- social_organization: cooperative_competitive

## w02_perfect_oracle: Perfect oracle world
Ground truth is directly and cheaply accessible to every agent.

- physics: stable
- logic: classical
- memory_reliability: perfect
- communication: perfect
- agent_lifespan: finite
- resource_availability: abundant
- observation_noise: none
- computation_cost: low
- learning_cost: low
- ground_truth_accessibility: direct
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: transparent
- time: linear
- social_organization: cooperative

## w03_heraclitean: Rule-chaos world
Rules change faster than evidence can accumulate.

- physics: unstable
- logic: classical
- memory_reliability: fallible
- communication: medium_bandwidth
- agent_lifespan: finite
- resource_availability: scarce
- observation_noise: high
- computation_cost: moderate
- learning_cost: high
- ground_truth_accessibility: indirect
- rule_stability: volatile
- causality: local
- identity_persistence: stable
- language: compositional
- time: linear
- social_organization: cooperative

## w04_false_memory: Adversarial memory world
Stored records drift or are adversarially corrupted more often than living observation.

- physics: stable
- logic: classical
- memory_reliability: adversarial
- communication: medium_bandwidth
- agent_lifespan: long
- resource_availability: moderate
- observation_noise: low
- computation_cost: moderate
- learning_cost: moderate
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: compositional
- time: linear
- social_organization: competitive

## w05_no_identity: No identity persistence world
Agents cannot preserve identity or commitments across time.

- physics: stable
- logic: classical
- memory_reliability: fallible
- communication: low_bandwidth
- agent_lifespan: instantaneous
- resource_availability: moderate
- observation_noise: moderate
- computation_cost: moderate
- learning_cost: high
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: none
- language: fragmentary
- time: linear
- social_organization: none

## w06_expensive_compute: Computation-starved world
Memory is cheap, but abstraction, search, and comparison are prohibitively expensive.

- physics: stable
- logic: classical
- memory_reliability: reliable
- communication: medium_bandwidth
- agent_lifespan: finite
- resource_availability: scarce
- observation_noise: moderate
- computation_cost: extreme
- learning_cost: high
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: compositional
- time: linear
- social_organization: cooperative

## w07_perfect_individual_minds: Perfect individual scientist world
Every agent has perfect private memory and unbiased reasoning but weak communication.

- physics: stable
- logic: classical
- memory_reliability: perfect_private
- communication: low_bandwidth
- agent_lifespan: long
- resource_availability: moderate
- observation_noise: low
- computation_cost: moderate
- learning_cost: moderate
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: idiosyncratic
- time: linear
- social_organization: individualist

## w08_total_transparency: Total transparency world
All observations and mental states are instantly public.

- physics: stable
- logic: classical
- memory_reliability: reliable
- communication: perfect
- agent_lifespan: finite
- resource_availability: moderate
- observation_noise: moderate
- computation_cost: moderate
- learning_cost: moderate
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: transparent
- time: linear
- social_organization: transparent_collective

## w09_hostile_social: Hostile prestige world
Criticism is mostly strategic sabotage, and preserved failures become reputational weapons.

- physics: stable
- logic: classical
- memory_reliability: reliable
- communication: medium_bandwidth
- agent_lifespan: finite
- resource_availability: scarce
- observation_noise: moderate
- computation_cost: moderate
- learning_cost: moderate
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: ambiguous
- time: linear
- social_organization: hostile_status

## w10_infinite_resources: Exhaustive search world
Resources are so abundant that brute-force enumeration dominates institutional design.

- physics: stable
- logic: classical
- memory_reliability: reliable
- communication: perfect
- agent_lifespan: long
- resource_availability: infinite
- observation_noise: low
- computation_cost: zero
- learning_cost: low
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: compositional
- time: linear
- social_organization: cooperative

## w11_noncompressible: Incompressible truth world
Truth is a lookup table with no reusable regularities.

- physics: algorithmically_random
- logic: classical
- memory_reliability: reliable
- communication: medium_bandwidth
- agent_lifespan: long
- resource_availability: moderate
- observation_noise: low
- computation_cost: moderate
- learning_cost: high
- ground_truth_accessibility: indirect
- rule_stability: stable_random
- causality: none
- identity_persistence: stable
- language: indexical
- time: linear
- social_organization: cooperative

## w12_single_observation: One-shot universe
Each phenomenon can be observed once and never repeated.

- physics: stable
- logic: classical
- memory_reliability: reliable
- communication: medium_bandwidth
- agent_lifespan: finite
- resource_availability: moderate
- observation_noise: moderate
- computation_cost: moderate
- learning_cost: high
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: stable
- language: compositional
- time: nonrepeatable
- social_organization: cooperative

## w13_nonmonotonic_logic: Nonmonotonic logic world
Valid inferences can be invalidated by later context in ways that resist stable compression.

- physics: contextual
- logic: nonmonotonic
- memory_reliability: reliable
- communication: medium_bandwidth
- agent_lifespan: long
- resource_availability: moderate
- observation_noise: moderate
- computation_cost: high
- learning_cost: high
- ground_truth_accessibility: indirect
- rule_stability: contextual
- causality: contextual
- identity_persistence: stable
- language: contextual
- time: linear
- social_organization: cooperative

## w14_many_worlds_personal_truth: Private-truth world
Each agent's observations are internally stable but not shared across agents.

- physics: observer_relative
- logic: paraconsistent
- memory_reliability: reliable
- communication: medium_bandwidth
- agent_lifespan: long
- resource_availability: moderate
- observation_noise: low
- computation_cost: moderate
- learning_cost: moderate
- ground_truth_accessibility: agent_local
- rule_stability: agent_local
- causality: observer_relative
- identity_persistence: stable
- language: partially_untranslatable
- time: linear
- social_organization: pluralist

## w15_immortal_hive: Immortal hive-mind world
There is one persistent collective mind with no internal disagreement.

- physics: stable
- logic: classical
- memory_reliability: perfect
- communication: identity
- agent_lifespan: immortal
- resource_availability: moderate
- observation_noise: moderate
- computation_cost: moderate
- learning_cost: moderate
- ground_truth_accessibility: indirect
- rule_stability: stable
- causality: local
- identity_persistence: collective
- language: internal
- time: linear
- social_organization: hive

## w16_deceptive_demon: Adversarial observation world
A powerful process adapts observations to defeat learning strategies.

- physics: adversarial
- logic: classical
- memory_reliability: reliable
- communication: medium_bandwidth
- agent_lifespan: finite
- resource_availability: scarce
- observation_noise: adversarial
- computation_cost: moderate
- learning_cost: extreme
- ground_truth_accessibility: blocked
- rule_stability: anti_inductive
- causality: adversarial
- identity_persistence: stable
- language: compositional
- time: linear
- social_organization: cooperative
