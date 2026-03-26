# Session Handoff

## Что это за проект

Это исследовательская ветка про когнитивную архитектуру, которая собирается из цепочки:

`inquiry -> belief -> revision -> synthesis -> archive/reactivation -> latent state -> causal latent -> hidden shift detection`

Проект не доказывает сознание. Он движется к более приземлённой и строгой цели:

`адаптивная эпистемическая машина, которая умеет спрашивать, проверять, менять объяснения мира и экономно управлять своими гипотезами.`

Сейчас внутри репозитория уже есть и отдельное прикладное ответвление:

- `epistemic_engine` — подпроект про выбор вопросов и пересмотр гипотез;
- первый MVP там — toy-debugging среда, где вопрос трактуется как диагностическое действие.

## Состояние репозитория

- GitHub `origin` уже подключён: `https://github.com/demidovz/maevtica.git`
- рабочая ветка сейчас `main`
- текущая git-идентичность для новых коммитов в этой репе: `Sergei <demidovz@yandex.ru>`

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

Текущее состояние прикладной линии `epistemic_engine`:

- на смешанном `meta-shift` benchmark лучшая общая политика сейчас `adaptive_shift / latent_shift`:
  - `accuracy 0.785`,
  - `mean_utility 0.150`.
- `latent_shift` не обогнал `adaptive_shift`, но вынес внутренний режим пересмотра в явный `BeliefState`:
  - `false_alarm_risk`,
  - `persistent_shift_risk`,
  - `switch_pressure`.
- `ArtifactDebuggingEnvironment` подтвердил перенос на semi-real артефакты:
  - `information_gain`, `type_memory`, `hybrid_memory` и `latent_shift` сейчас идут в паритете `1.000 / 1.900 / 0.552` (`accuracy / mean_cost / mean_utility`),
  - это полезный отрицательный результат: перенос уже есть, но библиотека кейсов пока слишком статична и слишком маленькая.

Текущий прикладной тезис:

`для хорошего выбора следующего вопроса в меняющемся мире системе мало просто держать вероятности гипотез;`
`ей нужен отдельный внутренний latent-state, который отслеживает риск ложной тревоги, риск устойчивой смены режима и силу переключения;`
`следующий переносной тест должен проверять это уже не в чисто synthetic мире, а в artifact-level shift / drift среде.`

## Текущие проценты

- Исследовательская линия: `85-87%`
- `epistemic_engine` MVP: `81/100`
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

Если нужна быстрая перепроверка текущей прикладной линии:

```powershell
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_debugging_meta_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --ambiguous-share 0.5 --false-alarm-length 1
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_artifact_debugging_benchmark.py --episodes 800 --seed 17
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\demo_debugging_meta_shift_mvp.py --seed 18 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --scenario auto --false-alarm-length 1
```

Если нужен общий обзор старых тестов:

```powershell
python C:\Users\user\workspace\maevtica\ideograph_experiments\run_all_experiments.py
```

## Что делать дальше

Следующие правильные технические шаги идут по двум линиям:

1. `ideograph_experiments`
   - добавить в recurrent state явный learned сигнал возврата режима вместо грубого override;
   - проверить `archive-match score`, `return-mode prior`, `mode reactivation confidence`;
   - учить probe-policy уже вместе с этим сигналом на long-horizon return-block benchmark;
   - снова сравнить `learned recurrent` против `hand-crafted risk-aware hidden-shift detector`.
2. `epistemic_engine`
   - сделать `artifact-level shift / drift` среду;
   - заставить внутри эпизода или между близкими инцидентами меняться полезность логов, diff, config и test-report источников;
   - проверить, даёт ли явный `shift_latent` выигрыш уже вне чисто synthetic world;
   - только после этого решать, нужен ли следующий шаг в сторону LLM/API или сначала ещё усложнять среду.

## Что важно не потерять

- Пользователь хочет не красивые графики, а чёткое утверждение или следующую гипотезу.
- После каждого теста нужен:
  - взрослый вывод,
  - перевод "по-детски",
  - следующая гипотеза,
  - её детский перевод.
- Удобный автономный режим: `4 итерации подряд`, потом остановка и отчёт.
- Отвечать всегда по-русски.
- В каждом сообщении пользователю отдельной строкой писать прогресс формата `До MVP: X/100`.
