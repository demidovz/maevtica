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
```

## Статус

- Исследовательская ясность: примерно `85-87%`
- Toy-архитектура / синтетический агент: примерно `55-60%`
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

Главный следующий шаг:

`learned switch-latent / return-mode signal`, а не только грубый high-risk revisit override.
