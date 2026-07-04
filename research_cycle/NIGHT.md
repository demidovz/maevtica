# Ночная вахта исследований — устав (2026-07-04)

Приказ босса (дословно, /goal 00:5x МСК): делать доделки и гнать ночной прогон;
крон каждые полчаса проверяет состояние; добавлять цели и вносить улучшения по
усмотрению; НИ В КОЕМ СЛУЧАЕ не останавливать процесс, пока недельный лимит
< 20%; если 5-часовая сессия подходит к 95% — подождать сброса и продолжить.

## Константы
- Репо: `/home/friemann/workspace/repos-demidovz/maevtica` (пуш в origin master).
- Петля: `research_cycle/cycle.workflow.js` (Workflow tool, scriptPath).
- Кампания казначея: `night-2026-07-04` (открыта на 14 пунктов: прибор 7% → стоп 21%).
- Прибор топлива: `~/workspace/maestratica/scripts/mst-usage --json` → `.week.pct`, `.session.pct`.
- Журналы вахты: `research_cycle/NIGHT_LOG.md` (тики), `research_cycle/RUNS.md` (запуски).

## Что делает каждый тик (крон, раз в 30 мин)
1. `mst-usage --json` → недельный % и 5ч %.
2. Записать строку в `NIGHT_LOG.md`: `HH:MM | неделя N% | 5ч M% | <решение>`.
3. **Если неделя ≥ 20%** → процесс можно сворачивать: дождаться конца текущего
   прогона, НЕ запускать новый, отметить в NIGHT_LOG «стоп по бюджету».
4. **Если 5ч ≥ 95%** → этот тик ничего не запускает (решение «пауза до сброса»);
   время сброса есть в `mst-usage` (`.session.resets`).
5. Иначе проверить живость прогона: свежесть файлов
   `research_cycle/campaigns/night-2026-07-04/round*.json` и последней записи RUNS.md.
   - Есть активность младше 45 мин → «работает, не трогаю».
   - Активности нет > 45 мин И в RUNS.md последний статус LAUNCHED →
     прогон умер: отметить STALLED, **перезапустить** (шаг ниже).
   - Последний статус DONE → запустить следующий прогон (шаг ниже).
6. Запуск прогона: Workflow tool,
   `scriptPath=/home/friemann/workspace/repos-demidovz/maevtica/research_cycle/cycle.workflow.js`,
   `args` = блок NEXT_ARGS ниже. После запуска дописать в RUNS.md:
   `ISO-время | LAUNCHED | <кратко args>`.
7. После завершившегося прогона (если это видно): закоммитить артефакты
   (`campaigns/`, `experiments/`, `RUNS.md`, `NIGHT_LOG.md`) в maevtica и запушить;
   собрать новый NEXT_ARGS: backlog = непроверенные (deferred) концепты из
   результата; domain тот же; capTokens 2500000; maxRounds 4; maxTest 2;
   controlEvery 2; campaign night-2026-07-04. Обновить блок NEXT_ARGS в этом файле.
8. Боссу ночью НЕ писать. Утренний доклад — задача главной сессии.

## Правила безопасности
- Не убивать чужие tmux/claude-сессии. Не трогать демонов студии.
- Опыты только на локальных моделях (gpt2/pythia-160m/opt-125m, venv
  `~/.local/state/mst/crc-venv311`), тестер time-boxed.
- Каждый прогон ограничен: maxRounds 4 + capTokens 2.5M + казначей (21% недели).

## NEXT_ARGS (обновляется после каждого прогона)
```json
{
  "domain": "mechanistic interpretability",
  "campaign": "night-2026-07-04",
  "maxRounds": 4,
  "maxTest": 2,
  "controlEvery": 2,
  "capTokens": 2500000,
  "backlog": [
    {
      "name": "Gated Feature (direction x context contract)",
      "definition": "A feature is a pair (v, g): direction v plus a sparse gate g — a predicate of <= 5 literals over co-occurring SAE-latent activity ('latent j active / inactive') — such that the mediation effect of patching along v conditional on g(x)=1 exceeds threshold theta, with g fit by decision tree / lasso on conditional-patching outcomes and validated on held-out inputs (precision >= 0.7). The four senses of 'feature' are predicted to coincide inside the gate and dissociate outside it; an ungated direction is not a feature, it is a marginal average over contexts.",
      "prediction": "For >= 25% of SAE latents whose unconditional mediation effect falls in the bottom quintile of their layer, there exists a learnable gate (<= 5 literals, held-out precision >= 0.7) under which their mediation effect exceeds the layer median by >= 3x — i.e., most 'causally dead but interpretable' latents are context-marginalization artifacts. Directly computable with conditional activation patching. Falsified if conditional patching rescues < 5% of dead latents.",
      "reduces_to": "Circuits / feature-interaction analysis (input-conditional feature effects in attribution graphs) — possibly just renaming 'a feature only matters inside its circuit' with a fitting procedure attached."
    }
  ],
  "seenStress": [
    "The word 'feature' itself: used simultaneously for (a) a direction in activation space, (b) an SAE dictionary latent, (c) a human-interpretable concept, (d) a causal mediator of behavior",
    "The word \"feature\" (Anthropic circuits program, SAE literature)"
  ]
}
```

Порядок очереди осознанный: κ (сама петля назвала главным кандидатом) → ρ → λ
(λ последняя — ей может не хватить локальных инструментов, честный исход
designed_not_run).
