# Epistemic Engine

Отдельный подпроект внутри `maevtica` про машину выбора вопросов и пересмотра гипотез.

Главная идея здесь такая:

`ответ не является центральным объектом;`
`центральный объект — следующий самый выгодный диагностический шаг.`

Проект строится вокруг цикла:

`гипотеза -> вопрос/действие -> наблюдение -> пересмотр -> следующий вопрос или действие`

## Зачем это вынесено отдельно

`ideograph_experiments` остаётся исследовательской линией про inquiry, belief dynamics, latent state и hidden shift.

`epistemic_engine` — более прикладная ветка: попытка собрать из этих идей отдельный минимальный движок, который умеет:

- держать несколько гипотез одновременно;
- выбирать следующий диагностический шаг;
- учитывать цену вопроса;
- пересматривать картину мира после наблюдения;
- останавливаться, когда уверенность уже достаточна.

## Первый MVP

Первый MVP здесь не про философию, а про `debugging / diagnosis`.

Теперь это уже не плоская среда с одной таблицей симптомов. Внутри неё есть:

- `6` конкурирующих причин бага;
- скрытый `surface profile` инцидента;
- дешёвые, но слабые проверки;
- более дорогие целевые probes, которые лучше разделяют гипотезы.

Возможные причины бага:

- `config_mismatch`
- `parser_bug`
- `dependency_drift`
- `state_leak`
- `schema_migration`
- `race_condition`

Система должна решить:

- что проверить следующим;
- сколько стоит эта проверка;
- насколько она разделяет гипотезы;
- когда уже можно остановиться и назвать наиболее вероятную причину.

Теперь benchmark идёт в `strict mode`:

- есть лимит на суммарную стоимость вопросов;
- есть лимит на число шагов;
- есть итоговый `utility score`, который штрафует за дорогой, затянутый или недоуверенный поиск.

Важный сдвиг:

`вопрос` здесь не обязательно текст.

Вопрос может быть действием:

- посмотреть дешёвые сигналы с дашборда;
- посмотреть лог ошибки;
- спросить пользователя о том, когда именно проявляется сбой;
- прочитать diff релиза;
- проверить конфиг;
- проверить lockfile;
- посмотреть код горячего пути;
- прогнать payload fixture;
- запустить узкий regression test;
- повторить сценарий в чистом процессе;
- проверить историю миграций;
- снять thread dump.

## Структура

`models.py`
- базовые структуры данных: belief state, observation, revision, benchmark result.

`beliefs/`
- базовые операции над вероятностями и выбор текущей верхней гипотезы.

`questions/`
- политика выбора следующего диагностического шага.

`revision/`
- байесовское обновление состояния и фиксация пересмотров.

`environments/`
- toy-среды.
- первая среда: `debugging.py`.
- внутри неё теперь есть и `DebuggingModeShiftEnvironment`, где режим мира может реально смениться внутри одного эпизода.

`policies/`
- baseline-политики: `information_gain`, `information_gain+memory`, `information_gain+type_memory`, `information_gain+mode_memory`, `information_gain+hybrid_memory`, `information_gain+switch_memory`, `cheapest`, `random`.

`memory/`
- простая case-memory, которая хранит похожие прошлые эпизоды и мягко подправляет выбор следующего шага.
- question-type memory, которая вспоминает не сам кейс, а класс следующего полезного вопроса.
- mode memory, которая хранит скрытые режимы инцидентов и мягко подправляет план поиска.
- hybrid memory, которая объединяет память о классе вопроса и память о режиме мира.
- switch-aware memory, которая поднимает отдельный сигнал: текущий remembered mode, похоже, начал расходиться с наблюдениями.

`benchmarks/`
- агрегирование метрик точности, стоимости и длины эпизода.

`runner/`
- пошаговое демо и сводный benchmark.

## Что уже можно запускать

Пошаговая демонстрация:

```powershell
python epistemic_engine\runner\demo_debugging_mvp.py --seed 11 --confidence-threshold 0.85
```

Сводный benchmark:

```powershell
python epistemic_engine\runner\run_debugging_benchmark.py --episodes 400 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5
```

Mode-shift benchmark:

```powershell
python epistemic_engine\runner\run_debugging_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5
```

Question-value-shift benchmark:

```powershell
python epistemic_engine\runner\run_debugging_value_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5
```

Ambiguous-shift benchmark:

```powershell
python epistemic_engine\runner\run_debugging_ambiguous_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --shift-probability 0.5 --false-alarm-length 1
```

Meta-shift benchmark:

```powershell
python epistemic_engine\runner\run_debugging_meta_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --ambiguous-share 0.5 --false-alarm-length 1
```

Artifact debugging benchmark:

```powershell
python epistemic_engine\runner\run_artifact_debugging_benchmark.py --episodes 800 --seed 17
```

Artifact question-value-shift benchmark:

```powershell
python epistemic_engine\runner\run_artifact_debugging_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --shift-after-step 2
```

Artifact ambiguous-shift benchmark:

```powershell
python epistemic_engine\runner\run_artifact_debugging_ambiguous_shift_benchmark.py --episodes 800 --seed 17 --confidence-threshold 0.85 --max-cost 6.0 --max-steps 5 --shift-after-step 2 --shift-probability 0.5 --false-alarm-length 1
```

## Что считаем успехом

На первом этапе успех — не “разумность вообще”, а более узкие свойства:

- меньше лишних диагностических шагов;
- ниже средняя стоимость поиска причины;
- выше точность финальной гипотезы;
- выше `mean_utility`;
- ниже доля остановок из-за лимита бюджета и лимита шагов;
- понятный и воспроизводимый пересмотр belief state.

## Текущий результат по памяти

Следующая гипотеза была такой:

`если система помнит похожие прошлые поломки, она будет быстрее выбирать полезный следующий шаг.`

Сейчас это реализовано в виде `information_gain+memory`.

На строгом benchmark это даёт слабый, но положительный эффект только в мягком режиме остановки:

- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain`: `accuracy 0.853`, `mean_utility 0.456`
  - `information_gain+memory`: `0.855`, `0.460`
- при `confidence>=0.85` память пока даёт почти паритет, а не явный выигрыш.

Текущий вывод:

`простая case-memory уже не ломает качество и иногда немного помогает;`
`но сильного скачка она пока не даёт;`
`память по целому похожему кейсу оказалась слабее, чем более структурированная память по типу вопроса или режиму мира.`

## Текущий результат по памяти классов вопросов

Следующая гипотеза была такой:

`если память вспоминает не целый прошлый кейс, а полезный тип следующего шага, это поможет лучше выбирать ход в похожей ситуации.`

Сейчас это реализовано в виде `information_gain+type_memory`.

На строгом benchmark эта версия уже сильнее, чем простая case-memory и базовый `information_gain`:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain`: `accuracy 0.863`, `mean_utility 0.421`
  - `information_gain+type_memory`: `0.871`, `0.432`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain`: `0.853`, `0.456`
  - `information_gain+type_memory`: `0.860`, `0.466`

Текущий вывод:

`память о классе следующего вопроса полезнее, чем память о целом похожем кейсе;`
`сильный эффект получается только при мягком бонусе, а не при грубом override;`
`вопрос теперь уже не в том, нужна ли структурированная память, а в том, какая именно её структура даёт лучший выигрыш.`

## Текущий результат по памяти режимов мира

Следующая гипотеза была такой:

`если система помнит не только тип полезного вопроса, но и скрытый режим мира, это поможет ей быстрее сужать круг причин.`

Сейчас это реализовано в виде `information_gain+mode_memory`.

На строгом benchmark эта версия лучше обычной case-memory и немного лучше базового `information_gain`, но пока всё ещё слабее `type_memory`:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain`: `accuracy 0.863`, `mean_utility 0.421`
  - `information_gain+mode_memory`: `0.865`, `0.427`
  - `information_gain+type_memory`: `0.871`, `0.432`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain`: `0.853`, `0.456`
  - `information_gain+mode_memory`: `0.856`, `0.462`
  - `information_gain+type_memory`: `0.860`, `0.466`

Текущий вывод:

`mode memory уже полезна;`
`она сильнее, чем память о целом похожем кейсе;`
`но в этой среде память о классе следующего вопроса пока даёт ещё лучший выигрыш;`
`следующий логичный шаг — собрать hybrid policy, где память о типе вопроса и память о режиме мира работают вместе, а не отдельно.`

## Текущий результат по гибридной памяти

Следующая гипотеза была такой:

`если память о типе вопроса и память о режиме мира работают вместе, они дадут лучший результат, чем каждая по отдельности.`

Сейчас это реализовано в виде `information_gain+hybrid_memory`.

На строгом benchmark гибрид действительно стал лучшей текущей политикой:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain`: `accuracy 0.863`, `mean_utility 0.421`
  - `information_gain+type_memory`: `0.871`, `0.432`
  - `information_gain+hybrid_memory`: `0.873`, `0.434`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain`: `0.853`, `0.456`
  - `information_gain+type_memory`: `0.860`, `0.466`
  - `information_gain+hybrid_memory`: `0.864`, `0.471`

Текущий вывод:

`гибридная память уже лучше, чем каждая из её частей по отдельности;`
`самый полезный вклад всё ещё идёт от памяти о типе следующего вопроса;`
`память о режиме мира даёт небольшой, но устойчивый дополнительный выигрыш при мягком весе;`
`следующий логичный шаг — добавить явное переключение между режимами мира, а не только мягкий prior из mode memory.`

## Текущий результат по switch-сигналу

Следующая гипотеза была такой:

`если система не только помнит режим мира, но и отдельно чувствует, что remembered mode начал ломаться, это даст ещё один выигрыш.`

Сейчас это реализовано в виде `information_gain+switch_memory`.

Результат смешанный:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain+hybrid_memory`: `accuracy 0.873`, `mean_utility 0.434`
  - `information_gain+switch_memory`: `0.871`, `0.434`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain+hybrid_memory`: `0.864`, `0.471`
  - `information_gain+switch_memory`: `0.863`, `0.466`

Текущий вывод:

`явный switch-сигнал уже полезен и не ломает политику;`
`но в текущей статической среде он пока не обгоняет лучший гибрид устойчиво;`
`скорее всего, его настоящий выигрыш проявится сильнее в среде, где режим может реально смениться внутри самого эпизода, а не только между эпизодами.`

## Текущий результат в mode-shift среде

Следующая гипотеза была такой:

`если внутри одного эпизода режим мира реально меняется, switch-сигнал наконец должен стать по-настоящему полезным.`

Сейчас это проверяется в `DebuggingModeShiftEnvironment`:

- сдвиг режима происходит внутри самого эпизода;
- benchmark не разрешает остановиться до того, как сдвиг успеет случиться;
- это уже не просто память о прошлых эпизодах, а проверка устойчивости к смене режима на ходу.

Результат:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain+hybrid_memory`: `accuracy 0.881`, `mean_utility 0.413`
  - `information_gain+switch_memory`: `0.879`, `0.421`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain+hybrid_memory`: `0.877`, `0.440`
  - `information_gain+switch_memory`: `0.873`, `0.435`

Текущий вывод:

`в среде с настоящим mode shift switch-сигнал наконец начинает окупаться;`
`в жёстком режиме он уже лучший по utility, хотя не лучший по accuracy;`
`в более мягком режиме лидер всё ещё hybrid memory;`
`значит switch-сигнал особенно полезен там, где цена позднего пересмотра выше, чем цена лишней осторожности.`

## Текущий результат в question-value-shift среде

Следующая гипотеза была такой:

`если внутри одного эпизода меняется не только профиль симптомов, но и то, какие типы вопросов вообще полезны, простая память о "хорошем следующем вопросе" должна просесть, а switch-aware policy должна стать явным лидером.`

Сейчас это проверяется в `DebuggingQuestionValueShiftEnvironment`:

- режим внутри эпизода по-прежнему сдвигается после второго наблюдения;
- вместе с режимом меняется сила типов вопросов: одни действия становятся информативнее, другие превращаются почти в шум;
- это уже проверка не только "заметил ли система сдвиг", но и "умеет ли она перестроить сам выбор класса следующего вопроса".

Результат:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain`: `accuracy 0.749`, `mean_utility 0.117`
  - `information_gain+hybrid_memory`: `0.755`, `0.121`
  - `information_gain+switch_memory`: `0.767`, `0.140`
  - `information_gain+switch_reactivation`: `0.755`, `0.121`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain`: `0.750`, `0.162`
  - `information_gain+hybrid_memory`: `0.754`, `0.169`
  - `information_gain+switch_memory`: `0.766`, `0.188`
  - `information_gain+switch_reactivation`: `0.752`, `0.168`

Текущий вывод:

`когда сдвиг меняет саму полезность типов вопросов, switch-aware memory уже становится явным лидером;`
`простая память о "какой тип вопроса обычно помогает" больше не доминирует, потому что вчерашний хороший вопрос после сдвига может стать плохим;`
`усложнённая switch_reactivation-политика в текущем виде не дала устойчивого выигрыша и пока выглядит как полезный отрицательный контроль, а не как новый лидер.`

## Текущий результат в ambiguous-shift среде

Следующая гипотеза была такой:

`если среда иногда даёт ложную тревогу, похожую на смену режима, полезно требовать подтверждение перед тем, как сильно перестраивать политику выбора вопросов.`

Сейчас это проверяется в `DebuggingAmbiguousShiftEnvironment`:

- часть эпизодов содержит настоящий question-value shift;
- часть эпизодов содержит только одиночный ложный сигнал и затем возвращается к старому режиму;
- benchmark не разрешает останавливаться раньше `4` шагов, чтобы у подтверждающей политики был шанс увидеть второй сигнал или его отсутствие.

Результат:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain`: `accuracy 0.751`, `mean_utility 0.079`
  - `information_gain+hybrid_memory`: `0.769`, `0.110`
  - `information_gain+switch_memory`: `0.776`, `0.117`
  - `information_gain+persistent_shift`: `0.776`, `0.125`
  - `information_gain+confirmed_switch`: `0.766`, `0.103`

Текущий вывод:

`смешанная среда оказалась полезной, потому что теперь мы честно тестируем настоящий сдвиг против ложной тревоги;`
`простое правило "подтверждай switch хотя бы двумя сигналами" не стало новым лидером;`
`а вот отдельный сигнал устойчивости сдвига уже полезен: persistent_shift догоняет switch_memory по accuracy и обходит его по utility;`
`при этом эта версия не универсальна: на чистой question-value-shift среде она хуже обычного switch_memory, потому что там нет цены за лишнюю осторожность к ложной тревоге;`
`значит линия теперь яснее: нужен не один глобальный switch-механизм, а более общий контроллер, который понимает, насколько среда склонна к one-off anomaly и persistent shift.`

## Текущий результат в meta-shift benchmark

Следующая гипотеза была такой:

`если не выбирать заранее одну жёсткую switch-политику, а смешивать агрессивный switch_memory и более осторожный persistent_shift по рисунку недавних сигналов, можно получить лучшую общую стратегию на смеси сред.`

Сейчас это проверяется в `run_debugging_meta_shift_benchmark.py`:

- половина эпизодов идёт из `ambiguous_shift`, где цена ложной тревоги реальна;
- половина идёт из `question_value_shift`, где важно не проспать настоящий устойчивый сдвиг;
- `adaptive_shift` пытается выбирать не один тип памяти навсегда, а режим пересмотра в зависимости от недавней динамики сигналов.

Результат:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain+hybrid_memory`: `accuracy 0.759`, `mean_utility 0.118`
  - `information_gain+switch_memory`: `0.782`, `0.149`
  - `information_gain+persistent_shift`: `0.766`, `0.131`
  - `information_gain+adaptive_shift`: `0.785`, `0.150`
  - `information_gain+latent_shift`: `0.785`, `0.150`

Текущий вывод:

`простая meta-идея уже проходит: adaptive_shift стал лучшей общей политикой на смешанном benchmark-е;`
`его выигрыш пока небольшой, но он важный по смыслу: система начала выбирать не только вопрос, но и режим собственного пересмотра;`
`latent_shift дал такой же результат, но вынес false_alarm_risk и persistent_shift_risk в явный внутренний state;`
`это важно не ради лишней красоты, а потому что теперь внутренний режим пересмотра живёт в BeliefState, обновляется пошагово через отдельный ShiftLatentUpdater и пишется в trace эпизода;`
`при этом adaptive_shift/latent_shift всё ещё не лучший в каждой отдельной среде по utility, поэтому это ещё не финальный контроллер, а первый рабочий шаг к нему.`

## Текущий результат в artifact debugging среде

Следующая гипотеза была такой:

`если перенести движок из чисто synthetic world в более похожую на реальность среду с логами, diff, config, lockfile и regression report, станет видно, что именно из текущей архитектуры уже переносится, а что пока завязано на игрушечные миры.`

Сейчас это проверяется в `ArtifactDebuggingEnvironment`:

- наблюдения здесь уже не абстрактные ярлыки, а короткие текстовые артефакты;
- runner по умолчанию идёт в жёстком режиме: `confidence>=0.95`, `max_cost=2.0`, `max_steps=3`;
- это пока не shift-среда, а скорее transfer sanity check для semi-real debugging cases.

Результат:

- при `800` эпизодах:
  - `information_gain`: `accuracy 1.000`, `mean_cost 1.900`, `mean_utility 0.552`
  - `information_gain+type_memory`: `1.000`, `1.900`, `0.552`
  - `information_gain+hybrid_memory`: `1.000`, `1.900`, `0.552`
  - `information_gain+latent_shift`: `1.000`, `1.900`, `0.552`
  - `cheapest`: `1.000`, `1.700`, `0.516`

Текущий вывод:

`перенос на semi-real artifacts уже есть: движок уверенно работает не только на символических outcome labels, но и на коротких логах, diff и test-report фразах;`
`но эта среда пока слишком статична и слишком маленькая, чтобы type memory, hybrid memory или latent shift начали реально выигрывать у обычного information gain;`
`значит следующий переносной шаг должен быть не "ещё один static case library", а artifact-level shift / drift, где сами полезные источники сигнала меняются внутри эпизода или между близкими инцидентами.`

## Текущий результат в artifact question-value-shift среде

Следующая гипотеза была такой:

`если внутри semi-real artifact эпизода меняется не диагноз, а полезность самих источников сигнала, shift-aware policy должна начать выигрывать уже вне чисто synthetic мира.`

Сейчас это проверяется в `ArtifactDebuggingQuestionValueShiftEnvironment`:

- внутри одного эпизода профиль сдвигается между `artifact_heavy` и `runtime_heavy`;
- до сдвига сильнее работают `history / inspect_artifact`, после сдвига — `inspect_code / run_test / probe_runtime`;
- benchmark не разрешает останавливаться раньше третьего шага, чтобы сдвиг успел реально повлиять на политику.

Результат:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain`: `accuracy 0.670`, `mean_utility -0.119`
  - `information_gain+hybrid_memory`: `0.670`, `-0.119`
  - `information_gain+adaptive_shift`: `0.679`, `-0.105`
  - `information_gain+latent_shift`: `0.679`, `-0.105`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain`: `0.669`, `-0.066`
  - `information_gain+hybrid_memory`: `0.669`, `-0.066`
  - `information_gain+adaptive_shift`: `0.679`, `-0.059`
  - `information_gain+latent_shift`: `0.679`, `-0.059`

Текущий вывод:

`artifact-level shift` уже проходит: shift-aware policy действительно начинает выигрывать на semi-real artifacts;`
`пока это не большой скачок, но уже честный перенос из synthetic мира в более похожую на жизнь диагностику;`
`latent_shift пока не обгоняет adaptive_shift, но уже воспроизводит его выигрыш в новой среде и делает внутренний режим пересмотра наблюдаемым.`

## Текущий результат в artifact ambiguous-shift среде

Следующая гипотеза была такой:

`если в том же semi-real artifact мире добавить ложные тревоги, станет видно, умеет ли shift-aware latent-state не только быстро реагировать на настоящий сдвиг, но и не переплачивать за одноразовую странность.`

Сейчас это проверяется в `ArtifactDebuggingAmbiguousShiftEnvironment`:

- часть эпизодов содержит настоящий устойчивый профильный сдвиг;
- часть содержит только короткий ложный `artifact -> runtime` сигнал и потом возвращается к исходному профилю;
- benchmark не разрешает останавливаться раньше `4` шагов.

Результат:

- при `confidence>=0.85`, `800` эпизодов:
  - `information_gain`: `accuracy 0.686`, `mean_utility -0.103`
  - `information_gain+hybrid_memory`: `0.686`, `-0.103`
  - `information_gain+persistent_shift`: `0.689`, `-0.083`
  - `information_gain+adaptive_shift`: `0.691`, `-0.099`
  - `information_gain+latent_shift`: `0.675`, `-0.086`
  - `information_gain+latent_trust_shift`: `0.675`, `-0.092`
- при `confidence>=0.8`, `800` эпизодов:
  - `information_gain`: `0.686`, `-0.066`
  - `information_gain+hybrid_memory`: `0.686`, `-0.066`
  - `information_gain+persistent_shift`: `0.689`, `-0.051`
  - `information_gain+adaptive_shift`: `0.691`, `-0.066`
  - текущая пересборка `latent_shift` на этой границе ещё не догонялась полным прогоном, поэтому опорной линией остаются цифры для `confidence>=0.85`

Текущий вывод:

`semi-real false alarms уже есть, и картина стала честнее;`
`adaptive_shift даёт лучшую accuracy, но по utility его обходит более осторожный persistent_shift;`
`текущий latent_shift уже стал чуть более cost-aware, чем adaptive_shift, но заплатил за это слишком большой просадкой accuracy;`
`latent_trust_shift` показал, что простого tempered observation update тоже недостаточно;`
`значит следующий шаг уже не в том, чтобы ещё раз доказать полезность shift-aware latent-state, а в том, чтобы разделить внутри него profile shift и hypothesis switch и только после этого снова калибровать false_alarm / persistent shift.`

## Текущий рабочий тезис

Сейчас тестируется такой тезис:

`для хорошего выбора следующего вопроса в меняющемся мире системе мало просто держать вероятности гипотез;`
`ей нужен отдельный внутренний latent-state, который отслеживает риск ложной тревоги, риск устойчивой смены режима и силу переключения;`
`artifact-level shift` и `artifact false alarm` уже пройдены;`
`следующий переносной шаг — сделать этот latent-state более cost-aware к one-off anomaly, не путая profile shift и hypothesis switch, чтобы он не уступал `persistent_shift` по utility.`

По-детски:

`штуке мало просто решать, какая догадка сейчас сильнее;`
`ей ещё нужно понимать, мир правда поменялся или просто на секунду показался странным;`
`теперь это надо проверить уже не в игрушечной игре, а в более похожей на жизнь поломке.`

Формат каждой следующей итерации теперь обязательный:

- текущий тезис;
- текущий тезис по-детски;
- следующий тезис;
- следующий тезис по-детски.

Если один из этих четырёх пунктов не зафиксирован явно, итерация считается оформленной не полностью.

## Следующий шаг

Следующий правильный эксперимент:

- улучшить semi-real `shift_latent`, чтобы он лучше различал `one-off anomaly` и `persistent shift`;
- отдельно откалибровать `false_alarm_risk` и `persistent_shift_risk` под artifact false alarms;
- снова сравнить `persistent_shift`, `adaptive_shift` и `latent_shift`;
- только потом решать, нужно ли подключать LLM/API или ещё развивать среду.

## Прогресс до MVP

- текущая оценка: `83/100`

## Куда это может расти

- в мета-слой над tool-using LLM;
- в движок выбора диагностических действий для coding/debugging;
- в исследовательского помощника;
- в сократическую обучающую систему;
- в слой управления гипотезами под дрейфующей средой.
