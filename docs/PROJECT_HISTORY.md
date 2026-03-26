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

## Текущая версия тезиса

Сейчас проект движется не к формуле "`сознание = граф`" и не к доказательству сознания.

Текущая версия такая:

`Интеллект — это управление вопросами, beliefs, latent states и пространством гипотез под давлением неопределённости, цены ошибки, скрытого контекста и смены режима мира.`

Следующий естественный шаг:

`learned switch-latent / return-mode-aware recurrent hidden belief state`
