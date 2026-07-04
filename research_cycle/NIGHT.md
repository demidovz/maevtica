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
  "runTag": "run3",
  "maxRounds": 2,
  "maxTest": 2,
  "controlEvery": 2,
  "capTokens": 1500000,
  "backlog": [
    {
      "name": "Causal Quotient Feature (intervention-equivalence class)",
      "definition": "Fix layer L and a behavior metric m (e.g. logit-diff on an eval set). Declare two perturbations equivalent, d1 ~ d2, iff max over eval contexts |m(h+d1) - m(h+d2)| <= epsilon. The representation of a concept is the equivalence class of the minimal-norm perturbation achieving a target behavioral shift, reported as (effective causal rank k = number of non-negligible singular values of the local perturbation-to-effect Jacobian, canonical representative = its top singular vector). 'Which direction IS the representation' becomes a category error: probe/steer/SAE vectors are different coset representatives of one quotient class.",
      "prediction": "Computable: for concepts where probe direction, difference-of-means steering vector, and matched SAE decoder row have pairwise cosine < 0.6 (the field's current embarrassment), >= 90% of the squared norm of their pairwise DIFFERENCE vectors lies in the causal null space (bottom singular subspace of the perturbation-effect Jacobian), and projecting all three onto the top-k causal subspace raises pairwise cosines to >= 0.9. FALSIFIED if the difference vectors carry substantial causal effect (i.e., the three objects genuinely do different things rather than being null-space-shifted copies).",
      "reduces_to": "Causal abstraction / DAS (Geiger et al.) and activation patching — risk it is causal mediation analysis wearing a quotient-space costume."
    },
    {
      "name": "Process-Pullback Geometry (shape law: the data process decides, not the model)",
      "definition": "The representation of a concept is the affine pullback into activation space of the belief-state coordinates of the minimal generative model (epsilon-machine / mixed-state presentation) of the relevant sublanguage of the training distribution: fit an affine map from activations to belief-simplex coordinates; the concept's geometry (line, circle, fractal, simplex face) is DEFINED as the image of the belief set under the inverse map. This converts the LRH from an hypothesis about models into a computable function of the data: linear features are exactly the concepts whose minimal process has a 1-simplex belief set.",
      "prediction": "Pre-registered shape forecasting: pick >= 5 structured domains where the mixed-state geometry is computable before touching the network (e.g. base-12 clock arithmetic -> circle; a 3-state Mess3-style HMM -> fractal; a binary flag process -> line). Predict, in advance, the intrinsic dimension and topology (Betti numbers via persistent homology) of the activation manifold a transformer trained on that domain will exhibit for the concept. Match rate must beat chance with p < 0.01. FALSIFIED by a single clean case of a model reaching the same next-token loss while representing a provably-circular process on a line (or vice versa) — that would show model-internal factors, not data geometry, set the shape.",
      "reduces_to": "Shai et al. belief-state fractal geometry itself — the only delta is promoting it from post-hoc explanation of one experiment to a forward-predictive definition of 'the representation'; if that promotion fails, it renamed nothing."
    }
  ],
  "seenStress": [
    "The word 'feature' itself: used simultaneously for (a) a direction in activation space, (b) an SAE dictionary latent, (c) a human-interpretable concept, (d) a causal mediator of behavior",
    "The word \"feature\" (Anthropic circuits program, SAE literature)",
    "Ablation/activation-patching as the field's causal ground truth — Hydra Effect (arXiv:2307.15771), Explorations of Self-Repair (arXiv:2402.15390)",
    "The Linear Representation Hypothesis / 'one concept = one direction' (Park et al. LRH; Arditi et al. 2024 'Refusal is mediated by a single direction'; Engels et al. 2024 'Not All Language Model Features Are Linear'; Wollschläger et al. 2025 'The Geometry of Refusal'; Shai et al. belief-state fractal geometry)"
  ]
}
```

Порядок очереди осознанный: κ (сама петля назвала главным кандидатом) → ρ → λ
(λ последняя — ей может не хватить локальных инструментов, честный исход
designed_not_run).
