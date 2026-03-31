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
- `ArtifactDebuggingQuestionValueShiftEnvironment` закрыл следующий переносной шаг:
  - внутри semi-real artifact эпизода теперь реально меняется полезность логов, diff, config, lockfile, code и runtime/test источников;
  - на этой среде `adaptive_shift / latent_shift` уже лучше голого `information_gain`:
    - при `confidence>=0.85`: `accuracy 0.679`, `mean_utility -0.105` против `0.670` и `-0.119`;
    - при `confidence>=0.8`: `0.679`, `-0.059` против `0.669`, `-0.066`;
  - `latent_shift` пока не обгоняет `adaptive_shift`, но уже воспроизводит его выигрыш на semi-real artifact shift, а не только в чисто synthetic мире.
- `ArtifactDebuggingAmbiguousShiftEnvironment` добавил в тот же semi-real мир ложные тревоги:
  - при `confidence>=0.85` `adaptive_shift` даёт лучшую `accuracy` (`0.691`), но лучшую `mean_utility` даёт `persistent_shift` (`-0.083`);
  - текущий `latent_shift` после belief-anchored / rebound-калибровки частично сдвинулся в правильную сторону по utility, но потерял слишком много accuracy:
    - `adaptive_shift`: `0.691 / -0.099`,
    - `latent_shift`: `0.675 / -0.086`,
    - `persistent_shift`: `0.689 / -0.083`;
  - эксперимент `latent_trust_shift`, где observation update темперируется через latent-state, не прошёл:
    - `0.675 / -0.092`;
  - при `confidence>=0.8` картина та же:
    - `adaptive_shift / latent_shift`: `0.691 / -0.066`,
    - `persistent_shift`: `0.689 / -0.051`;
  - это важное уточнение: на semi-real false alarms текущий `latent_shift` уже не проваливается, но и не становится лучшим cost-aware контроллером.

Текущий прикладной тезис:

`для хорошего выбора следующего вопроса в меняющемся мире системе мало просто держать вероятности гипотез;`
`ей нужен отдельный внутренний latent-state, который отслеживает риск ложной тревоги, риск устойчивой смены режима и силу переключения;`
`artifact-level shift` и `artifact-level false alarm` уже пройдены;`
`теперь вопрос уже не в том, нужен ли такой latent-state вообще, а в том, как разделить внутри него profile shift и hypothesis switch и сделать его более cost-aware на ложных тревогах, чтобы он не уступал `persistent_shift` по utility.`

## Текущие проценты

- Исследовательская линия: `85-87%`
- `epistemic_engine` MVP: `83/100`
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
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_artifact_debugging_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --shift-after-step 2
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_artifact_debugging_ambiguous_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --shift-after-step 2 --shift-probability 0.5 --false-alarm-length 1
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\demo_debugging_meta_shift_mvp.py --seed 18 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --scenario auto --false-alarm-length 1
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\demo_artifact_debugging_shift_mvp.py --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --shift-after-step 2
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\demo_artifact_debugging_ambiguous_shift_mvp.py --seed 18 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --shift-after-step 2 --false-alarm-length 1 --scenario auto
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
   - улучшить `shift_latent` под semi-real false alarms;
   - явно закодировать различие между `one-off anomaly`, `profile shift` и `hypothesis switch` так, чтобы не проигрывать `persistent_shift` по utility;
   - потом снова сравнить `persistent_shift`, `adaptive_shift` и `latent_shift` уже на artifact ambiguous-shift benchmark;
   - только после этого решать, нужен ли следующий шаг в сторону LLM/API или сначала ещё усложнять среду.

## Что важно не потерять

- Пользователь хочет не красивые графики, а чёткое утверждение или следующую гипотезу.
- После каждого теста нужен:
  - взрослый вывод,
  - перевод "по-детски",
  - следующая гипотеза,
  - её детский перевод.
- В каждой итерации нужно отдельно фиксировать:
  - текущий тезис, который проверяется сейчас,
  - его перевод "по-детски",
  - следующий тезис / следующую гипотезу,
  - её перевод "по-детски".
- Не считать это необязательной риторикой: без этого считается, что итерация оформлена не до конца.
- Удобный автономный режим: `4 итерации подряд`, потом остановка и отчёт.
- Отвечать всегда по-русски.
- В каждом сообщении пользователю отдельной строкой писать прогресс формата `До MVP: X/100`.
