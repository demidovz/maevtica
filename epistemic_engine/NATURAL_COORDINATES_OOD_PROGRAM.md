# Natural Coordinates OOD Validation Program

This program treats the discovered 2D Koopman quadratic affine coordinate
system as a frozen scientific theory.

Rule 0:

- no coordinate refit on OOD data;
- no latent dimension change for the frozen theory;
- no coordinate optimization;
- no OOD parameter tuning;
- blind prediction before scoring.

The frozen coordinate map is loaded from:

`epistemic_engine/outputs/natural_coordinates_validation/frozen_coordinate_system.json`

The affine law is fit once on the reference discovery trajectory family, then
stored and used unchanged on all OOD worlds.

Run:

```bash
python -m epistemic_engine.runner.run_natural_coordinates_ood --cases-per-family 6 --steps 10 --seed 2026
```

Main outputs:

- `ood_benchmark_suite.json`
- `prediction_accuracy_distribution.csv`
- `failure_atlas.json`
- `applicability_boundary.json`
- `competing_model_comparison.json`
- `robustness_report.json`
- `confidence_intervals.json`
- `scientific_assessment.json`

