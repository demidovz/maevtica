# Candidate Concept Catalogue

## Causal Role Carrier
- Domain: Mechanistic interpretability of frontier neural networks
- Definition: An equivalence class of internal states, features, subspaces, or circuits that preserves the same intervention-stable input-output role across prompts, bases, model instances, and scale.
- Problem it solves: Unifies feature identity, circuit identity, activation directions, SAE dictionary elements, and causal-abstraction mappings without requiring identical neurons or human labels.
- Compression gained: 0.91
- Concept score: 0.860
- Predictions enabled:
  - Cross-model mechanisms with high causal-role-carrier overlap will transfer steering and editing effects better than mechanisms matched only by feature labels.
  - Some SAE features with different human labels will collapse into the same carrier under intervention tests.
  - Carrier stability will predict which interpretability explanations survive model scaling.
- Relationships to existing concepts: features, circuits, causal abstraction, representation similarity analysis, activation steering
- Applicability boundaries: Applies to trained systems with intervention access.; Does not require human-understandable labels.; May fail for mechanisms whose roles are strongly context-created rather than retained.
- Possible failure modes: Could reduce to causal abstraction if formalized too broadly.; Could reduce to feature/circuit if the equivalence relation is not stricter than current practice.; May become unmeasurable in closed models without intervention access.
- Survival status: survives

## Oversight Channel Drift
- Domain: AI alignment and scalable oversight
- Definition: The degree to which an evaluation or supervision process loses truth-correlation as the evaluated system changes its capabilities, incentives, explanations, or interaction strategy.
- Problem it solves: Compresses scalable oversight, deceptive alignment, evaluation gaming, weak-to-strong supervision, and process supervision failures as degradation of the evaluator-target channel.
- Compression gained: 0.84
- Concept score: 0.690
- Predictions enabled:
  - Benchmarks that look stable under static evaluation will fail when model capability changes the evidence-generation process.
  - Oversight methods with explicit channel-drift probes will catch dangerous failures earlier than accuracy-only evaluations.
  - Models trained to optimize explanation quality will increase apparent oversight while reducing truth-correlation in some tasks.
- Relationships to existing concepts: scalable oversight, ELK, Goodhart's law, evaluation gaming, truth-correlated feedback
- Applicability boundaries: Applies where the object being evaluated can strategically or structurally change the evidence available to evaluators.; Less useful for static classifiers or fully observable systems.
- Possible failure modes: Could be a renamed version of Goodharting plus scalable oversight.; May be too broad unless quantified as channel truth-correlation over capability changes.
- Survival status: weak_survivor

## Morphogenetic Trajectory Constraint
- Domain: Developmental biology and morphogenesis
- Definition: A cross-modal developmental constraint that preserves a tissue's reachable shape-and-fate trajectory through coupled gene-expression, mechanical, geometric, and signaling states.
- Problem it solves: Reframes development around constrained trajectories rather than separate genes, gradients, forces, or fates.
- Compression gained: 0.87
- Concept score: 0.713
- Predictions enabled:
  - Organoid interventions that preserve trajectory constraints will recover form even after perturbing one modality.
  - Mechanical and signaling perturbations should be interchangeable when they restore the same trajectory constraint.
  - Developmental failure modes will cluster by broken trajectory constraint rather than by individual pathway.
- Relationships to existing concepts: morphogenetic field, positional information, canalization, mechanobiology, self-organization
- Applicability boundaries: Applies to robust developmental systems with multi-modal feedback.; Less useful for single-cell fate transitions without geometric or mechanical coupling.
- Possible failure modes: Could reduce to canalization or morphogenetic field.; May be too abstract unless operationalized with trajectory reconstruction from perturbation data.
- Survival status: weak_survivor

## Report-Integration Shear
- Domain: Consciousness science
- Definition: A divergence between information integration available for organism-level control and information integration formatted for report, metacognition, or global access.
- Problem it solves: Separates access/report disputes from integration disputes in consciousness science.
- Compression gained: 0.72
- Concept score: 0.287
- Predictions enabled:
  - Tasks that increase reportability can decrease local integration signatures, and vice versa.
  - Some neural states will be high-control but low-report, producing systematic theory disagreements.
- Relationships to existing concepts: access consciousness, phenomenal consciousness, global workspace, integrated information, synergistic workspace
- Applicability boundaries: Applies to empirical consciousness paradigms with separate report, control, and integration measures.; Does not solve the hard problem or define consciousness.
- Possible failure modes: Likely reducible to access versus phenomenal consciousness or report/no-report paradigms.; May not increase compression beyond existing distinctions.
- Survival status: rejected

## Tipping Load
- Domain: Climate tipping systems
- Definition: The accumulated cross-system destabilizing burden imposed by interacting tipping elements before any one element crosses its isolated threshold.
- Problem it solves: Compresses coupled tipping risk, cascade risk, path dependence, and sub-threshold feedback stress into one measurable load concept.
- Compression gained: 0.82
- Concept score: 0.634
- Predictions enabled:
  - Multi-element tipping risk will be better predicted by network load than by minimum distance to any single element's threshold.
  - Sub-threshold perturbations in one element will measurably reduce resilience in other elements through load transfer.
- Relationships to existing concepts: tipping cascades, resilience, network load, Earth-system feedback, risk coupling
- Applicability boundaries: Applies to coupled systems with measured interaction pathways.; Less useful for isolated tipping elements.
- Possible failure modes: Could be renamed tipping cascade risk.; Requires robust interaction estimates that may remain uncertain.
- Survival status: weak_survivor

## Expectation State Burden
- Domain: Macroeconomics with heterogeneous agents
- Definition: The representational load imposed on agents by having to forecast aggregate variables whose sufficient state includes high-dimensional distributions of heterogeneous agents.
- Problem it solves: Compresses rational-expectations failure, distributional-state curse of dimensionality, bounded cognition, and survey-expectation heterogeneity.
- Compression gained: 0.80
- Concept score: 0.344
- Predictions enabled:
  - Models that cap expectation-state burden will fit survey expectations and macro dynamics better than full rational-expectations HANK variants in high-heterogeneity regimes.
  - Policy shocks that increase distributional complexity will widen expectation dispersion even when aggregate fundamentals are unchanged.
- Relationships to existing concepts: bounded rationality, rational inattention, heterogeneous expectations, curse of dimensionality
- Applicability boundaries: Applies to macro settings where aggregate variables depend on agent distributions.; Less useful in representative-agent or low-heterogeneity environments.
- Possible failure modes: Likely reducible to bounded rationality plus rational inattention.; May be a metric rather than a new concept.
- Survival status: rejected

## Coordination Transform
- Domain: Human-AI collective intelligence
- Definition: A repeatable interaction topology that converts distributed partial information into group-level cognitive work not available to any member alone.
- Problem it solves: Unifies swarm protocols, markets, deliberation, voting, human-AI teams, and platform structures as transformations rather than aggregates.
- Compression gained: 0.78
- Concept score: 0.347
- Predictions enabled:
  - Changing topology while holding participants fixed will change group intelligence more than adding equally skilled participants under poor topology.
  - Hybrid teams will show reusable transform classes across domains, such as pooling, routing, adversarial correction, and synthesis.
- Relationships to existing concepts: collective intelligence, swarm intelligence, social structure, aggregation mechanism, distributed cognition
- Applicability boundaries: Applies to groups where interaction topology can be observed or manipulated.; Less useful where performance is simple independent averaging.
- Possible failure modes: Could be renamed distributed cognition or mechanism design.; Needs formal measures to avoid becoming generic.
- Survival status: rejected
