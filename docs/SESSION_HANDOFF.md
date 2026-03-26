# Session Handoff

## Что это за проект

Это исследовательская ветка про когнитивную архитектуру, которая собирается из цепочки:

`inquiry -> belief -> revision -> synthesis -> archive/reactivation -> latent state -> causal latent -> hidden shift detection`

Проект не доказывает сознание. Он движется к более приземлённой и строгой цели:

`адаптивная эпистемическая машина, которая умеет спрашивать, проверять, менять объяснения мира и экономно управлять своими гипотезами.`

## Где проект сейчас

Сильные уже подтверждённые результаты:

- Вопрос работает как оператор снижения неопределённости.
- Цена вопроса порождает экономный inquiry.
- Память снижает повторные вопросы и улучшает перенос.
- Без grounding система уходит в псевдосемантику.
- Beliefs становятся устойчивыми, когда замкнуты на действие и ошибку.
- Конфликт beliefs может вести к синтезу нового belief.
- Синтезируемые beliefs могут переноситься, но перенос должен быть селективным.
- Архив, pruning и reactivation реально нужны.
- Иерархический routing по family / subfamily / rule / package / mode даёт выигрыш.
- Hidden context можно выводить из неполных наблюдений.
- Compressed latent state лучше плоской памяти по сигнатурам.
- `learned_inquiry_policy_test.py` показал, что policy может выучить выгодный уровень вопроса и переучиться после drift.
- `causal_latent_intervention_test.py` показал, что причинный latent держится там, где observational shortcut разваливается.
- `long_horizon_integrated_agent_test.py` показал, что integrated agent уже может долго жить в меняющемся мире.
- `long_horizon_hidden_shift_test.py` показал, что скрытую смену режима можно ловить без внешнего drift hint.

Главный текущий блокер:

- `learned_hidden_shift_batch_suite.py` показал, что простой обучаемый detector по короткому локальному state пока нестабилен.

Это сужает тезис:

`простой contextual bandit gate недостаточен;`
`нужен richer hidden belief state или recurrent/model-based detector.`

## Текущие проценты

- Исследовательская линия: `85-87%`
- Toy-архитектура: `55-60%`
- Конечная сильная цель: `35-40%`

## Что запускать в первую очередь

Если нужна быстрая перепроверка последних опорных результатов:

```powershell
python C:\Users\user\workspace\maevtica\ideograph_experiments\learned_inquiry_policy_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\causal_latent_intervention_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_integrated_agent_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_hidden_shift_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\learned_hidden_shift_batch_suite.py
```

Если нужен общий обзор старых тестов:

```powershell
python C:\Users\user\workspace\maevtica\ideograph_experiments\run_all_experiments.py
```

## Что делать дальше

Следующий правильный технический шаг:

1. Сделать `recurrent_hidden_shift_detector_test.py` или близкий по смыслу эксперимент.
2. Дать агенту внутреннее скрытое состояние, которое обновляется по последовательности ошибок и наблюдений, а не по одному локальному bucket state.
3. Только после этого снова сравнивать `learned` против `hand-crafted risk-aware hidden-shift detector`.

## Что важно не потерять

- Пользователь хочет не красивые графики, а чёткое утверждение или следующую гипотезу.
- После каждого теста нужен:
  - взрослый вывод,
  - перевод "по-детски",
  - следующая гипотеза,
  - её детский перевод.
- Удобный автономный режим: `4 итерации подряд`, потом остановка и отчёт.
- Отвечать всегда по-русски.
