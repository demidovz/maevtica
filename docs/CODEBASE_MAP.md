# Codebase Map

## Корень проекта

- [ideograph_research.md](C:\Users\user\workspace\maevtica\ideograph_research.md)
  Основной исследовательский текст, из которого выросла серия экспериментов.
- [README.md](C:\Users\user\workspace\maevtica\README.md)
  Краткий обзор проекта.
- [docs/SESSION_HANDOFF.md](C:\Users\user\workspace\maevtica\docs\SESSION_HANDOFF.md)
  Быстрый handoff для новой сессии.
- [docs/PROJECT_HISTORY.md](C:\Users\user\workspace\maevtica\docs\PROJECT_HISTORY.md)
  История эволюции тезиса и цепочки тестов.
- [docs/WORKING_WITH_USER.md](C:\Users\user\workspace\maevtica\docs\WORKING_WITH_USER.md)
  Правила взаимодействия с пользователем.

## Основной каталог экспериментов

Сейчас в [ideograph_experiments](C:\Users\user\workspace\maevtica\ideograph_experiments) `36` Python-файлов и примерно `14k` строк.

### Базовые тесты inquiry / uncertainty

- [question_uncertainty_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\question_uncertainty_test.py)
- [intrinsic_reward_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\intrinsic_reward_test.py)
- [memory_transfer_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\memory_transfer_test.py)
- [grounding_semantics_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\grounding_semantics_test.py)
- [self_reinforcing_symbolism_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\self_reinforcing_symbolism_test.py)

### Belief dynamics

- [belief_action_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_action_test.py)
- [belief_revision_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_revision_test.py)
- [belief_conflict_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_conflict_test.py)
- [belief_synthesis_transfer_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_synthesis_transfer_test.py)
- [belief_synthesis_pruning_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_synthesis_pruning_test.py)
- [belief_pruning_reactivation_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_pruning_reactivation_test.py)

### Transfer / gating / diagnostics

- [belief_transfer_discrimination_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_transfer_discrimination_test.py)
- [belief_transfer_gating_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_transfer_gating_test.py)
- [belief_transfer_diagnostic_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_transfer_diagnostic_test.py)

### Archive / composition / class switching

- [belief_archive_selection_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_archive_selection_test.py)
- [belief_archive_composition_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_archive_composition_test.py)
- [belief_archive_class_selection_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_archive_class_selection_test.py)
- [belief_class_switch_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_class_switch_test.py)
- [belief_class_switch_diagnostic_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_class_switch_diagnostic_test.py)

### Hierarchy / budget / batch suites

- [belief_class_switch_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_class_switch_batch_suite.py)
- [belief_hierarchy_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_hierarchy_batch_suite.py)
- [belief_hierarchy_budget_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_hierarchy_budget_batch_suite.py)
- [belief_hierarchy_dynamic_budget_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_hierarchy_dynamic_budget_batch_suite.py)
- [belief_meta_budget_learning_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_meta_budget_learning_batch_suite.py)

### Hidden context / latent state

- [belief_hidden_context_inference_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_hidden_context_inference_batch_suite.py)
- [belief_latent_compression_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\belief_latent_compression_batch_suite.py)
- [causal_latent_intervention_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\causal_latent_intervention_test.py)

### Long horizon / learned policies

- [learned_inquiry_policy_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\learned_inquiry_policy_test.py)
- [long_horizon_integrated_agent_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_integrated_agent_test.py)
- [long_horizon_hidden_shift_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_hidden_shift_test.py)
- [learned_hidden_shift_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\learned_hidden_shift_batch_suite.py)

### Сервисный раннер

- [run_all_experiments.py](C:\Users\user\workspace\maevtica\ideograph_experiments\run_all_experiments.py)
  Сводный запуск старой серии экспериментов и генерация summary/plots.

## Что считать опорными файлами

Если нужно быстро восстановить контекст на новой машине:

1. [ideograph_research.md](C:\Users\user\workspace\maevtica\ideograph_research.md)
2. [docs/SESSION_HANDOFF.md](C:\Users\user\workspace\maevtica\docs\SESSION_HANDOFF.md)
3. [docs/PROJECT_HISTORY.md](C:\Users\user\workspace\maevtica\docs\PROJECT_HISTORY.md)
4. [learned_inquiry_policy_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\learned_inquiry_policy_test.py)
5. [causal_latent_intervention_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\causal_latent_intervention_test.py)
6. [long_horizon_integrated_agent_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_integrated_agent_test.py)
7. [long_horizon_hidden_shift_test.py](C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_hidden_shift_test.py)
8. [learned_hidden_shift_batch_suite.py](C:\Users\user\workspace\maevtica\ideograph_experiments\learned_hidden_shift_batch_suite.py)
