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

## Текущая версия тезиса

Сейчас проект движется не к формуле "`сознание = граф`" и не к доказательству сознания.

Текущая версия такая:

`Интеллект — это управление вопросами, beliefs, latent states и пространством гипотез под давлением неопределённости, цены ошибки, скрытого контекста и смены режима мира.`

Следующий естественный шаг:

`recurrent / model-based hidden belief state`
