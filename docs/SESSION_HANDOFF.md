# Session Handoff

## Что это за проект

Это исследовательская ветка про когнитивную архитектуру, которая собирается из цепочки:

`inquiry -> belief -> revision -> synthesis -> archive/reactivation -> latent state -> causal latent -> hidden shift detection`

Проект не доказывает сознание. Он движется к более приземлённой и строгой цели:

`адаптивная эпистемическая машина, которая умеет спрашивать, проверять, менять объяснения мира и экономно управлять своими гипотезами.`

Сейчас внутри репозитория уже есть и отдельное прикладное ответвление:

- `epistemic_engine` — подпроект про выбор вопросов и пересмотр гипотез;
- первый MVP там — toy-debugging среда, где вопрос трактуется как диагностическое действие.

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
- `recurrent_hidden_shift_detector_test.py` показал, что richer recurrent hidden belief state уже лучше flat learned contextual gate на mixed hidden-shift eval:
  - mean reward `0.792` против `0.759`,
  - floor по сценариям `0.752` против `0.699`.
- `return_mode_signal_test.py` показал, что явный `return-mode signal` поверх лучшей learned версии даёт ещё небольшой выигрыш на long horizon:
  - `iter4`: `0.892`,
  - `iter5`: `0.897`,
  - hand-crafted baseline: `0.913`.

Главный текущий блокер:

- `learned_hidden_shift_batch_suite.py` показал, что простой обучаемый detector по короткому локальному state пока нестабилен.
- `recurrent_hidden_shift_detector_test.py` показал, что recurrent hidden belief state помогает, но всё ещё не догоняет hand-crafted `risk_aware_hidden_shift` на long horizon:
  - `0.796` против `0.853`,
  - probe rate пока слишком низкий (`0.031` против `0.125`).
- `return_mode_signal_test.py` показал, что простой явный сигнал возврата режима помогает, но только частично:
  - long horizon стал лучше,
  - mixed качество не выросло,
  - зазор до hand-crafted baseline остался.

Это сужает тезис:

`простой contextual bandit gate недостаточен;`
`recurrent hidden belief state уже даёт выигрыш;`
`грубый явный return-mode signal уже помогает;`
`следующий недостающий кусок — более точный learned switch-latent / return-mode signal вместо простого override.`

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
python C:\Users\user\workspace\maevtica\ideograph_experiments\recurrent_hidden_shift_detector_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\return_mode_signal_test.py
```

Если нужен общий обзор старых тестов:

```powershell
python C:\Users\user\workspace\maevtica\ideograph_experiments\run_all_experiments.py
```

## Что делать дальше

Следующий правильный технический шаг:

1. Добавить в recurrent state явный сигнал возврата режима:
   - не как грубый override,
   - а как полноценный learned latent.
2. Проверить варианты:
   - `archive-match score`,
   - `return-mode prior`,
   - `mode reactivation confidence`.
3. Учить probe-policy уже вместе с этим сигналом на long-horizon return-block benchmark.
4. После этого снова сравнивать `learned recurrent` против `hand-crafted risk-aware hidden-shift detector`.

## Что важно не потерять

- Пользователь хочет не красивые графики, а чёткое утверждение или следующую гипотезу.
- После каждого теста нужен:
  - взрослый вывод,
  - перевод "по-детски",
  - следующая гипотеза,
  - её детский перевод.
- Удобный автономный режим: `4 итерации подряд`, потом остановка и отчёт.
- Отвечать всегда по-русски.
