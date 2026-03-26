# Maevtica

Исследовательский репозиторий по линии `ideograph / belief dynamics`.

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

Ключевые команды:

```powershell
python C:\Users\user\workspace\maevtica\ideograph_experiments\learned_inquiry_policy_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\causal_latent_intervention_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_integrated_agent_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\long_horizon_hidden_shift_test.py
python C:\Users\user\workspace\maevtica\ideograph_experiments\learned_hidden_shift_batch_suite.py
```

## Статус

- Исследовательская ясность: примерно `85-87%`
- Toy-архитектура / синтетический агент: примерно `55-60%`
- Сильная конечная цель: примерно `35-40%`

Главный текущий результат:

- hand-crafted `risk-aware hidden-shift detector` проходит;
- наивный `learned hidden-shift detector` на грубом локальном state пока не проходит стабильно.

Главный следующий шаг:

`recurrent / model-based hidden belief state`, а не ещё один плоский gate по короткому снимку состояния.
