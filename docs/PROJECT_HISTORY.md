# Project History

## Исходная точка

Стартовая интуиция была такой:

`интеллект начинается с вопроса`

Из [ideograph_research.md](C:\Users\user\workspace\maevtica\ideograph_research.md) выросла последовательная исследовательская программа. Дальше проект шёл не как одна большая модель, а как серия узких Python-тестов, каждый из которых либо усиливал тезис, либо отрезал слишком сильную формулировку.

## Как развивался тезис

### 1. Вопрос как снижение неопределённости

- `question_uncertainty_test.py`
- `intrinsic_reward_test.py`

Вывод:

`вопрос` важен не как языковая форма, а как действие, уменьшающее неопределённость. Полезный inquiry должен быть экономным и учитывать цену вопроса.

### 2. Память и grounding

- `memory_transfer_test.py`
- `grounding_semantics_test.py`
- `self_reinforcing_symbolism_test.py`

Вывод:

Память полезна только в grounded-мире. Без внешней проверки система уходит в ложную уверенность и замкнутую псевдосемантику.

### 3. Belief как цикл вопроса, действия и ошибки

- `belief_action_test.py`
- `belief_revision_test.py`

Вывод:

Belief требует связи с действием и штрафом за ошибку. Полезный belief должен обладать управляемой инерцией.

### 4. Конфликт, синтез и перенос

- `belief_conflict_test.py`
- `belief_synthesis_transfer_test.py`
- `belief_transfer_discrimination_test.py`
- `belief_transfer_gating_test.py`
- `belief_transfer_diagnostic_test.py`

Вывод:

Одной конкуренции beliefs мало. При систематическом конфликте полезная система умеет синтезировать новый belief. Перенос должен быть селективным, а применимость переноса стоит проверять диагностическими вопросами.

### 5. Иерархия режимов и архив

- `belief_archive_selection_test.py`
- `belief_archive_composition_test.py`
- `belief_archive_class_selection_test.py`
- `belief_pruning_reactivation_test.py`
- `belief_synthesis_pruning_test.py`

Вывод:

Нужны archive, pruning, reactivation и выбор не только belief, но и класса объяснения. Иногда новый режим можно собрать из старых частей, а иногда нужен новый принцип.

### 6. Иерархический routing и budget control

- `belief_hierarchy_batch_suite.py`
- `belief_hierarchy_budget_batch_suite.py`
- `belief_hierarchy_dynamic_budget_batch_suite.py`
- `belief_meta_budget_learning_batch_suite.py`

Вывод:

Система должна выбирать уровень вопроса и тратить бюджет вопросов на правильный уровень в правильный момент. Но метаполитика вопроса должна быть контекстной.

### 7. Hidden context и latent compression

- `belief_hidden_context_inference_batch_suite.py`
- `belief_latent_compression_batch_suite.py`

Вывод:

Система должна уметь выводить скрытый контекст и сжимать множество наблюдаемых сигнатур в компактные latent states. Простая память по случаям хуже, чем latent compression с возможностью refine/split/merge.

### 8. Learned inquiry, causal latent и long horizon

- `learned_inquiry_policy_test.py`
- `causal_latent_intervention_test.py`
- `long_horizon_integrated_agent_test.py`
- `long_horizon_hidden_shift_test.py`

Вывод:

- policy может выучить, когда и на каком уровне спрашивать;
- causal latent лучше observational shortcut;
- integrated agent уже может долго жить в меняющемся мире;
- hand-crafted risk-aware hidden-shift detector работает и без внешнего drift hint.

### 9. Где линия споткнулась

- `learned_hidden_shift_batch_suite.py`

Вывод:

`learned hidden-shift detector` в простой форме пока не даёт устойчивой победы. Это важный отрицательный результат.

Текущая корректировка тезиса:

`недостаточно научить систему решать probe/act по короткому локальному state;`
`нужно учить richer hidden belief state по последовательности событий.`

### 10. Recurrent hidden belief state

- `recurrent_hidden_shift_detector_test.py`

Вывод:

- более богатое recurrent hidden state действительно улучшает learned hidden-shift detector;
- на mixed eval по `10` seed-ам recurrent версия бьёт flat contextual baseline:
  - `0.792` против `0.759` по среднему reward,
  - `0.752` против `0.699` по худшему сценарию;
- но сильная версия гипотезы пока не проходит:
  - recurrent learned detector всё ещё слабее hand-crafted `risk_aware_hidden_shift` на long horizon (`0.796` против `0.853`).

Новая корректировка тезиса:

`просто richer recurrent belief state уже полезен;`
`но без явного switch-latent / return-mode signal learned detector всё ещё недобирает на возвратах режимов и на политике probe.`

### 11. Explicit return-mode signal

- `return_mode_signal_test.py`

Вывод:

- простой явный `return-mode signal` действительно даёт ещё один шаг вперёд;
- версия `iter5` улучшает long horizon относительно `iter4`:
  - `0.897` против `0.892`;
- но это пока слабое улучшение, не закрывающее зазор до hand-crafted `risk_aware_hidden_shift` (`0.913`).

Новая корректировка тезиса:

`явный return-mode signal нужен;`
`но грубого high-risk revisit override недостаточно;`
`следующий шаг — сделать этот сигнал learned и встроенным в hidden belief state.`

### 12. Выделение отдельного подпроекта

- `epistemic_engine/`

Вывод:

- из исследовательской линии начал выделяться отдельный прикладной трек;
- его цель уже не объяснять всё сразу, а собрать минимальную машину выбора вопросов и пересмотра гипотез;
- первый MVP намеренно узкий: toy-debugging / diagnosis вместо общей философской платформы.

Новая развилка проекта:

`ideograph_experiments` остаётся местом, где проверяются базовые гипотезы про inquiry, belief и hidden shift;`
`epistemic_engine` становится местом, где эти идеи собираются в отдельный движок с понятным интерфейсом, политиками и benchmark-ами.`

### 13. Shift-aware контроллер в `epistemic_engine`

- `run_debugging_meta_shift_benchmark.py`
- `epistemic_engine/policies/switch_memory.py`
- `epistemic_engine/beliefs/shift_latent.py`

Вывод:

- простого `information_gain` уже недостаточно, когда среда может давать и настоящие сдвиги, и ложные тревоги;
- `hybrid_memory`, `switch_memory`, `persistent_shift` и `adaptive_shift` показали, что системе полезно выбирать не только следующий вопрос, но и режим собственного пересмотра;
- `latent_shift` не улучшил метрики относительно `adaptive_shift`, но сделал внутренний механизм явным и наблюдаемым:
  - `false_alarm_risk`,
  - `persistent_shift_risk`,
  - `switch_pressure`;
- на смешанном `meta-shift` benchmark лучшая общая политика сейчас `adaptive_shift / latent_shift`:
  - `accuracy 0.785`,
  - `mean_utility 0.150`.

Новая корректировка тезиса:

`для хорошего выбора следующего вопроса в меняющемся мире мало просто держать вероятности гипотез;`
`нужен отдельный внутренний latent-state, который отслеживает риск ложной тревоги, риск устойчивой смены режима и силу переключения.`

### 14. Перенос на semi-real artifact debugging

- `epistemic_engine/environments/artifact_debugging.py`
- `epistemic_engine/runner/run_artifact_debugging_benchmark.py`

Вывод:

- перенос с чисто synthetic outcome labels на короткие логи, diff, config, lockfile и test-report артефакты уже получился;
- `information_gain`, `type_memory`, `hybrid_memory` и `latent_shift` в текущей artifact-среде идут в паритете:
  - `accuracy 1.000`,
  - `mean_cost 1.900`,
  - `mean_utility 0.552`;
- это не победа сложной памяти, а полезный отрицательный результат:
  - текущая библиотека кейсов пока слишком маленькая и статичная,
  - поэтому перенос уже есть, но следующий архитектурный выигрыш пока не проявляется.

Новая корректировка тезиса:

`следующий переносной тест должен быть не про ещё один static case library;`
`он должен быть про artifact-level shift / drift, где полезность источников сигнала меняется внутри эпизода или между близкими инцидентами.`

### 15. Artifact-level question-value shift

- `epistemic_engine/environments/artifact_debugging.py`
- `epistemic_engine/runner/run_artifact_debugging_shift_benchmark.py`

Вывод:

- следующий переносной шаг действительно оказался правильным: теперь внутри semi-real artifact эпизода меняется не диагноз, а полезность самих источников сигнала;
- `information_gain` и `hybrid_memory` на новой среде идут в паритете, а `adaptive_shift / latent_shift` уже оказываются лучше:
  - при `confidence>=0.85`: `0.679 / -0.105` против `0.670 / -0.119`,
  - при `confidence>=0.8`: `0.679 / -0.059` против `0.669 / -0.066`
  в формате `accuracy / mean_utility`;
- `latent_shift` пока не обгоняет `adaptive_shift`, но теперь его архитектурный смысл уже подтверждён не только в synthetic shift benchmark-ах, но и в semi-real artifact shift среде.

Новая корректировка тезиса:

`artifact-level shift` уже проходит;`
`следующий вопрос уже не в том, нужен ли shift-aware latent-state вообще, а в том, помогает ли он отличать настоящий semi-real shift от ложной тревоги.`

### 16. Artifact-level ambiguous shift / false alarm

- `epistemic_engine/environments/artifact_debugging.py`
- `epistemic_engine/runner/run_artifact_debugging_ambiguous_shift_benchmark.py`

Вывод:

- следующий переносной шаг тоже пройден: теперь semi-real artifact среда умеет давать не только настоящий сдвиг профиля, но и короткую ложную тревогу с возвратом;
- на этой среде `adaptive_shift / latent_shift` дают лучшую `accuracy`, но не лучшую `utility`;
- `persistent_shift` оказывается сильнее именно по cost-aware критерию:
  - при `confidence>=0.85`: `adaptive_shift = 0.691 / -0.099`, `latent_shift = 0.675 / -0.086`, `persistent_shift = 0.689 / -0.083`,
  - при `confidence>=0.8`: `0.691 / -0.066` против `0.689 / -0.051`
  в формате `accuracy / mean_utility`;
- отдельные попытки улучшить semi-real latent через action robustness и tempered observation update не прошли:
  - `latent_trust_shift` дал `0.675 / -0.092` и не обошёл ни `latent_shift`, ни `persistent_shift`;
- это уже не отрицание `shift_latent`, а более узкое уточнение:
  - текущий semi-real latent-state уже полезен,
  - но пока ещё недостаточно cost-aware в мире с ложными тревогами.

Новая корректировка тезиса:

`нужен не просто shift-aware latent-state;`
`нужен latent-state, который не путает profile shift и hypothesis switch, лучше различает one-off anomaly и persistent shift и поэтому не проигрывает осторожной policy по utility.`

## Текущая версия тезиса

Сейчас проект движется не к формуле "`сознание = граф`" и не к доказательству сознания.

Текущая версия такая:

`Интеллект — это управление вопросами, beliefs, latent states и пространством гипотез под давлением неопределённости, цены ошибки, скрытого контекста и смены режима мира.`

Внутри неё сейчас уже две рабочие ветки:

- `ideograph_experiments`: `learned switch-latent / return-mode-aware recurrent hidden belief state`
- `epistemic_engine`: `cost-aware semi-real shift_latent`, который лучше различает `one-off anomaly` и `persistent shift`
