export const meta = {
  name: 'maevtica-research-cycle',
  description: 'Falsification-first research loop with teeth: shine a flashlight into one meaning-space, generate concepts, try to kill them (adversary + prior-art + real external test), keep survivors, stop at the campaign budget, emit an honest report.',
  whenToUse: 'Run one budgeted research campaign over a domain. Set args.capTokens from research_cycle/treasurer.py; the loop self-stops at the cap.',
  phases: [
    { title: 'Explore',  detail: 'find representation stress in the domain' },
    { title: 'Generate', detail: 'propose candidate concepts with testable predictions' },
    { title: 'Attack',   detail: 'adversary + historian try to kill each' },
    { title: 'Test',     detail: 'real external experiment for testable survivors' },
    { title: 'Report',   detail: 'honest narrative of searched / found / refuted / survived' },
  ],
}

// args: { domain: string, capTokens?: number, maxRounds?: number, controlEvery?: number }
const DOMAIN = (args && args.domain) || 'mechanistic interpretability'
const MAX_ROUNDS = (args && args.maxRounds) || 6
const CONTROL_EVERY = (args && args.controlEvery) || 3
const DRY = !!(args && args.dryTest)  // shakedown: tester designs only, runs nothing
const RESERVE = 0.12  // stop with ~12% of the slice unspent, for the report itself
// The token cap from the казначей (treasurer.py). budget.spent() ALWAYS works
// (unlike budget.total, which is null without a turn directive — the bug the
// 2026-07-03 shakedown caught: the cap silently didn't apply and the loop ran
// free for 5 rounds / 51 agents). Gate on spent() so the cap is always enforced.
const CAP = (args && args.capTokens) || null
const underBudget = () => !CAP || budget.spent() < CAP * (1 - RESERVE)

// ── structured-output schemas ───────────────────────────────────────────────
const S_STRESS = { type: 'object', required: ['stress_points'], properties: { stress_points: { type: 'array', items: { type: 'object', required: ['where', 'why', 'testable'], properties: {
  where: { type: 'string' }, why: { type: 'string', description: 'what strains: exceptions pile up, predictions fail, concepts overloaded' },
  testable: { type: 'boolean', description: 'can a concept here make a prediction checkable by code/experiment?' } } } } } }
const S_CANDS = { type: 'object', required: ['candidates'], properties: { candidates: { type: 'array', items: { type: 'object', required: ['name', 'definition', 'prediction', 'reduces_to'], properties: {
  name: { type: 'string' }, definition: { type: 'string', description: 'OPERATIONAL — measurable, not poetic' },
  prediction: { type: 'string', description: 'a concrete, falsifiable prediction; ideally computable' },
  reduces_to: { type: 'string', description: 'nearest existing concept it might just be renaming' } } } } } }
const S_ADV = { type: 'object', required: ['survives', 'counterexample', 'is_rename'], properties: {
  survives: { type: 'boolean' }, counterexample: { type: 'string', description: 'concrete case that breaks it, or "none found"' },
  is_rename: { type: 'boolean', description: 'true if it is just existing concept X under a new name' }, reason: { type: 'string' } } }
const S_HIST = { type: 'object', required: ['is_rediscovery', 'prior_art'], properties: {
  is_rediscovery: { type: 'boolean' }, prior_art: { type: 'array', items: { type: 'string' }, description: 'named prior concepts/fields covering this' }, novelty: { type: 'string' } } }
const S_TEST = { type: 'object', required: ['status', 'verdict'], properties: {
  status: { type: 'string', enum: ['ran', 'designed_not_run', 'not_computable'] },
  verdict: { type: 'string', enum: ['supported', 'refuted', 'inconclusive', 'n/a'] },
  design: { type: 'string', description: 'the falsification design + preregistered decision rule' },
  numbers: { type: 'string', description: 'key measured numbers, or why none' },
  sanity: { type: 'string', description: 'oracle/positive-control check — did the measurement even work? (guards the .norm-bug class of error)' } } }
const S_CONTROL = { type: 'object', required: ['judges_trustworthy'], properties: {
  judges_trustworthy: { type: 'boolean', description: 'did known-good concepts pass and planted distractors get killed?' }, notes: { type: 'string' } } }

// ── stage prompts (falsification-first; encode today's lessons) ──────────────
const P = {
  explorer: (seen) => `You are the EXPLORER for domain: "${DOMAIN}". Find REPRESENTATION STRESS — places where the field's current concepts strain: exceptions pile up, predictions fail, one word is overloaded with many meanings, or practitioners keep patching. Prefer stress points where a new concept could make a prediction checkable by a real experiment. Do NOT repeat already-scanned points: ${JSON.stringify(seen).slice(0, 1500)}. Return the schema.`,
  generator: (stress, killed) => `You are the GENERATOR for "${DOMAIN}". Given this stress point: ${JSON.stringify(stress)}. Propose 2-4 candidate NEW representations that would relieve it. Each MUST have (a) an OPERATIONAL definition (measurable, not poetry), (b) a concrete FALSIFIABLE prediction (ideally computable), (c) the nearest existing concept it might just be renaming. Avoid ideas already killed: ${JSON.stringify(killed).slice(0, 1200)}. Facts are cheap, representations are expensive — only propose what earns its keep.`,
  adversary: (c) => `You are the ADVERSARY. Try HARD to KILL this concept, do not be charitable: "${c.name}" — ${c.definition}. Prediction: ${c.prediction}. Find a concrete counterexample. Decide: is it just "${c.reduces_to}" renamed? Default to survives=false if you cannot find a real, non-trivial, hard-to-replace distinction. A concept survives only if it compresses multiple phenomena AND makes a distinctive prediction AND has clear failure boundaries.`,
  historian: (c) => `You are the HISTORIAN. Has science already got "${c.name}" (${c.definition}) under another name? Search prior art across relevant fields (information theory, RL, statistics, the domain's own literature, adjacent disciplines). Benchmark-008 lesson: most "new" ideas are rediscoveries — bias toward is_rediscovery=true unless the novelty is specific and defensible.`,
  tester: (c) => `You are the TESTER — the TEETH. Concept: "${c.name}", prediction: "${c.prediction}". ${DRY ? 'SHAKEDOWN MODE: DESIGN ONLY — do NOT run anything, do NOT touch Bash or models. Return status=designed_not_run with the falsification design + preregistered decision rule + the oracle/positive-control you WOULD check.' : 'If the prediction is computable, DESIGN a minimal falsification (preregister a decision rule BEFORE running), then run it with Bash on local open models (template + protocol in research_cycle/experiments/; venv ~/.local/state/mst/crc-venv311). MANDATORY sanity: report an oracle / positive-control number proving the measurement actually worked — a near-zero oracle means the run is broken, NOT that the concept failed (this exact check caught a .norm bug on 2026-07-03).'} Be honest: "survived my test" ≠ "proven". If not computable in-budget, status=designed_not_run or not_computable with the design written down.`,
  control: () => `You are the CONTROL. Calibrate the judges: take 2 concepts KNOWN to be real (e.g. entropy, gene) and 2 obvious DISTRACTORS (vague/renamed) for "${DOMAIN}", run them through the same adversary+historian bar this campaign uses, and report whether the known-good passed and the distractors were killed. If the judges can't tell them apart, the campaign's verdicts are untrustworthy.`,
  report: (journal) => `You are the REPORTER. Write a BEAUTIFUL, HONEST report for a non-technical boss from this campaign journal: ${JSON.stringify(journal).slice(0, 12000)}. Structure: (1) where we shone the flashlight and why; (2) the interesting thoughts we found; (3) what we tried to build and REFUTED, and how (this is the valuable part — most pans are empty, say so plainly); (4) the survivors, with honest caveats ("survived our tests" ≠ "proven"); (5) what a next campaign should chase. No jargon-dumping; put technical detail under a "детали по запросу" tail. Register: the reader is a boss, not a co-developer.`,
}

// ── the loop ────────────────────────────────────────────────────────────────
const journal = { domain: DOMAIN, seenStress: [], rounds: [], survivors: [], killed: [], controls: [] }
let round = 0
log(`flashlight → "${DOMAIN}" · cap≈${(CAP || 0).toLocaleString()} output tokens · maxRounds ${MAX_ROUNDS} · reserve ${RESERVE}`)

while (round < MAX_ROUNDS && underBudget()) {
  round++
  phase('Explore')
  const stress = await agent(P.explorer(journal.seenStress), { label: `explore#${round}`, phase: 'Explore', schema: S_STRESS })
  if (!stress || !stress.stress_points || !stress.stress_points.length) { log(`round ${round}: no fresh stress found — flashlight exhausted`); break }
  const top = stress.stress_points.sort((a, b) => (b.testable === a.testable ? 0 : b.testable ? 1 : -1))[0]
  journal.seenStress.push(top.where)

  phase('Generate')
  const gen = await agent(P.generator(top, journal.killed), { label: `generate#${round}`, phase: 'Generate', schema: S_CANDS })
  const cands = (gen && gen.candidates) || []

  phase('Attack')
  const judged = (await parallel(cands.map(c => () =>
    parallel([
      () => agent(P.adversary(c), { label: `adv:${c.name}`, phase: 'Attack', schema: S_ADV }),
      () => agent(P.historian(c), { label: `hist:${c.name}`, phase: 'Attack', schema: S_HIST }),
    ]).then(([adv, hist]) => ({ c, adv, hist }))
  ))).filter(Boolean)

  const survivors = [], killed = []
  for (const j of judged) {
    const ok = j.adv && j.adv.survives && !j.adv.is_rename && j.hist && !j.hist.is_rediscovery
    if (ok) survivors.push(j); else killed.push({ name: j.c.name, why: (j.adv && j.adv.counterexample) || (j.hist && j.hist.prior_art) })
  }
  journal.killed.push(...killed.map(k => k.name))

  phase('Test')
  const tested = (await parallel(survivors.map(s => () =>
    agent(P.tester(s.c), { label: `test:${s.c.name}`, phase: 'Test', schema: S_TEST, effort: 'high' })
      .then(t => ({ name: s.c.name, concept: s.c, test: t }))
  ))).filter(Boolean)

  // a survivor is only "kept" if its test didn't refute it
  for (const t of tested) if (!t.test || t.test.verdict !== 'refuted') journal.survivors.push(t)

  journal.rounds.push({ round, stress: top, generated: cands.length, killed, tested })
  log(`round ${round}: ${cands.length} proposed · ${killed.length} killed · ${tested.length} tested · ${journal.survivors.length} survivors so far`)

  if (round % CONTROL_EVERY === 0) {
    const ctl = await agent(P.control(), { label: `control#${round}`, phase: 'Attack', schema: S_CONTROL })
    journal.controls.push(ctl)
    if (ctl && ctl.judges_trustworthy === false) { log(`round ${round}: CONTROL FAILED — judges can't tell real from distractor; halting to avoid self-fooling`); break }
  }
}

phase('Report')
const report = await agent(P.report(journal), { label: 'report', phase: 'Report', effort: 'high' })
log(`campaign done · ${round} rounds · ${journal.survivors.length} survivors · ${journal.killed.length} killed`)
return { domain: DOMAIN, rounds: round, survivors: journal.survivors, killed: journal.killed, controls: journal.controls, journal, report }
