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
      "name": "Coding sparsity (native firing statistics of a property)",
      "definition": "For a probed property P, take its best linear-probe direction d_P and compute the distribution of d_P·activation over a token corpus. Coding sparsity κ(P) = heavy-tailedness of that distribution (excess kurtosis, or Gini/L0 of thresholded activations, or 1 − firing-rate). High κ = P fires rarely with heavy tails (matches an SAE's L1/L0 sparsity prior); low κ = P is dense/near-Gaussian (mismatches the prior, so the SAE splits or absorbs it). Measurable from probes alone, before training the SAE.",
      "prediction": "SAE-atom recovery of P — max cosine between any trained atom and d_P — increases monotonically with κ(P). Concrete computable split: properties below median κ get max-cosine <0.5 (SAE misses them, exactly the class that 'fails to beat probes' in arXiv:2502.16681); properties above median κ get max-cosine >0.8. Falsified if recovery is flat or non-monotone in κ.",
      "reduces_to": "Superposition sparsity + feature splitting/absorption (Chanin et al. 2024). Risk: 'how sparse is the property's activation' renamed. Earns keep as an a-priori per-property scalar that predicts SAE recoverability instead of a post-hoc failure label."
    },
    {
      "name": "Carrier rank (representational dimensionality of a property)",
      "definition": "For property P, carrier rank ρ(P) = number of orthogonal probe directions needed to reach 95% of the maximum achievable probe accuracy, measured by iterative nullspace projection (INLP) or the participation ratio of between-class mean differences. ρ=1 means P genuinely lives on a single direction (the 'one direction' assumption of SAEs holds); ρ>1 means P is a subspace/polytope that no single atom can capture. Computed from probes, before the SAE.",
      "prediction": "The count of SAE atoms significantly aligned with P (cosine>0.3) grows with ρ(P), while single-atom max-cosine recovery falls as ρ grows. Concrete: ρ=1 properties have exactly one dominant matching atom (cos>0.8); ρ≥3 properties have no atom above 0.5 and instead spread across ≥ρ atoms (measured feature-splitting count ≈ ρ). Falsified if split-count is independent of ρ, or if high-ρ properties are still captured by a single atom.",
      "reduces_to": "Feature splitting and multi-dimensional/manifold features (Engels et al. 2024). Risk of renaming 'feature splitting'. Earns keep as an a-priori geometric scalar predicting how many atoms a property fragments into."
    },
    {
      "name": "Legibility coordinate (auto-interp fidelity as its own axis)",
      "definition": "Legibility λ(atom) = simulate-and-score fidelity of the atom's natural-language label: correlation (or F1) between activations predicted from the label and true activations (Bills/EleutherAI auto-interp protocol). The ~38% auto-interp failure rate = fraction with λ below threshold. The representational move: treat λ as a coordinate ORTHOGONAL to read-fraction and coding sparsity — it isolates human-nameability from the causal and sparsity senses of 'feature'.",
      "prediction": "Across atoms, λ is near-independent of read-fraction R and of coding sparsity κ: predict |Pearson r| < 0.2 for both. Corollaries: (i) among top-decile most-causal atoms a substantial share are illegible ('dark computation'); (ii) among the most legible atoms a substantial share are causally inert. Falsified if λ correlates strongly (|r|>0.5) with R or κ. NB: needs label simulation — if no cheap local path exists, designed_not_run with the design written down is the honest outcome.",
      "reduces_to": "Auto-interp / explanation scoring (Bills et al.). Only new content is the asserted orthogonality to the causal and sparsity axes."
    }
  ]
}
```

Порядок очереди осознанный: κ (сама петля назвала главным кандидатом) → ρ → λ
(λ последняя — ей может не хватить локальных инструментов, честный исход
designed_not_run).
