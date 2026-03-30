# Maevtica

Исследовательский репозиторий по линии `ideograph / belief dynamics`.

Сейчас внутри репозитория уже две отдельные рабочие ветки:

- `ideograph_experiments` — основная исследовательская серия тестов про inquiry, belief dynamics, latent state и hidden shift.
- `epistemic_engine` — отдельный подпроект про машину выбора вопросов и пересмотра гипотез, начиная с toy-debugging MVP.

Главная идея проекта больше не формулируется как "`интеллект начинается с вопроса`" в наивном виде. Текущая рабочая формулировка строже:

`Интеллект — это управление вопросами, beliefs, latent states и пространством гипотез под давлением неопределённости, цены ошибки, скрытого контекста и смены режимов мира.`

Сейчас в репозитории основная активная часть — каталог [ideograph_experiments](C:\Users\user\workspace\maevtica\ideograph_experiments) с последовательностью Python-экспериментов.

## Быстрый старт

Ключевые файлы для новой сессии:

- [ideograph_research.md](C:\Users\user\workspace\maevtica\ideograph_research.md)
- [docs/SESSION_HANDOFF.md](C:\Users\user\workspace\maevtica\docs\SESSION_HANDOFF.md)
- [docs/PROJECT_HISTORY.md](C:\Users\user\workspace\maevtica\docs\PROJECT_HISTORY.md)
- [docs/CODEBASE_MAP.md](C:\Users\user\workspace\maevtica\docs\CODEBASE_MAP.md)
- [docs/WORKING_WITH_USER.md](C:\Users\user\workspace\maevtica\docs\WORKING_WITH_USER.md)
- [epistemic_engine/README.md](C:\Users\user\workspace\maevtica\epistemic_engine\README.md)

Ключевые команды:

```powershell
python C:\Users\user\workspace\maevtica\ideograph_experiments\learned_inquiry_policy_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\causal_latent_intervention_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_integrated_agent_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_hidden_shift_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\learned_hidden_shift_batch_suite.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\recurrent_hidden_shift_detector_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\return_mode_signal_test.py
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\demo_debugging_mvp.py
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_debugging_benchmark.py
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_debugging_meta_shift_benchmark.py
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_artifact_debugging_benchmark.py
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_artifact_debugging_shift_benchmark.py
python C:\Users\user\workspace\maevtica\epistemic_engine\runner\run_artifact_debugging_ambiguous_shift_benchmark.py
```

## Статус

- Исследовательская ясность: примерно `85-87%`
- `epistemic_engine` MVP: примерно `83/100`
- Сильная конечная цель: примерно `35-40%`

Главный текущий результат:

- hand-crafted `risk-aware hidden-shift detector` проходит;
- наивный `learned hidden-shift detector` на грубом локальном state пока не проходит стабильно;
- `recurrent_hidden_shift_detector_test.py` показал, что richer recurrent hidden belief state уже бьёт flat learned contextual gate на mixed eval:
  - mean reward `0.792` против `0.759`,
  - floor по сценариям `0.752` против `0.699`,
  - но на long horizon всё ещё слабее hand-crafted `risk-aware hidden-shift detector` (`0.796` против `0.853`).
- `return_mode_signal_test.py` показал, что явный `return-mode signal` поверх `iter4` версии даёт небольшой дополнительный выигрыш на long horizon:
  - `0.897` против `0.892`,
  - но до hand-crafted baseline `0.913` всё ещё есть заметный зазор.
- в `epistemic_engine` лучшая общая политика на смешанном `meta-shift` benchmark сейчас `adaptive_shift / latent_shift`:
  - `accuracy 0.785`,
  - `mean_utility 0.150`.
- `ArtifactDebuggingEnvironment` показал, что движок уже переносится на semi-real logs / diff / config / test-report артефакты, но текущая библиотека кейсов пока слишком статична и слишком лёгкая, чтобы память или latent-state давали дополнительный выигрыш.
- `ArtifactDebuggingQuestionValueShiftEnvironment` показал, что после добавления `artifact-level shift` `adaptive_shift / latent_shift` уже лучше голого `information_gain` и на semi-real artifacts:
  - при `confidence>=0.85`: `0.679 / -0.105` против `0.670 / -0.119`,
  - при `confidence>=0.8`: `0.679 / -0.059` против `0.669 / -0.066`
  в формате `accuracy / mean_utility`.
- `ArtifactDebuggingAmbiguousShiftEnvironment` добавил ложные тревоги в semi-real artifact-мир и показал более честную границу:
  - при `confidence>=0.85` лучшая `accuracy` у `adaptive_shift` (`0.691`), но лучшая `mean_utility` у `persistent_shift` (`-0.083`);
  - текущий `latent_shift` после belief-anchored / rebound-калибровки стал чуть более cost-aware, чем `adaptive_shift`, но заплатил за это точностью:
    - `adaptive_shift`: `0.691 / -0.099`,
    - `latent_shift`: `0.675 / -0.086`,
    - `persistent_shift`: `0.689 / -0.083`;
  - экспериментальный `latent_trust_shift` с tempered observation update не прошёл:
    - `0.675 / -0.092`.

Главные следующие шаги:

- `ideograph_experiments`: `learned switch-latent / return-mode signal`, а не только грубый high-risk revisit override.
- `epistemic_engine`: улучшить semi-real `shift_latent`, чтобы он не путал `profile shift` и `hypothesis switch`, лучше различал `one-off anomaly` и `persistent shift` и перестал проигрывать `persistent_shift` по utility на artifact false alarms.
