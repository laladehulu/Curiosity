# GEPA-Pareto vs Behavior-Archive: A Workshop-Scope Ablation

## Scope and venue

**Target:** A CoRL 2026 workshop submission (any of the workshops accepting
work on LLM-RL, quality-diversity, or reward design). Workshops typically
accept 4-6 page papers with pilot-scale evidence and fast (3-4 week) review.

**Hardware:** Mac CPU only (M-series Apple Silicon assumed). No CUDA. No
Isaac Gym. Tasks chosen for CPU tractability.

**Honest framing the human author should preserve in the paper:**

- "Pilot ablation on classic control benchmarks"
- "We do not claim generalization to high-DoF dexterous manipulation"
- "Single-task analyses with [N] seeds; results indicative, not definitive"

Workshop reviewers tolerate this honesty. Main-track reviewers wouldn't.

---

## The question

GEPA (Agrawal et al., 2025) selects reward candidates using a Pareto frontier
over **per-instance task scores**. Two reward functions survive selection if
they cover different task instances well, regardless of whether the policies
they produce behave similarly.

MAP-Elites-style selection (Mouret & Clune, 2015) maintains an archive over
**behavioral descriptors of resulting policies**. Two reward functions survive
if they produce behaviorally distinct policies, regardless of task-instance
coverage.

These are different diversity mechanisms operating on the same population.
Nobody has directly compared them for LLM-based reward design.

**Central question:** When LLMs generate reward candidates for robot RL,
does behavior-archive selection produce a *qualitatively different* and
*downstream-more-useful* set of survivors than GEPA's task-instance Pareto?

---

## What this paper claims and does not claim

**Claims (achievable at workshop scale):**

1. On classic control tasks of varying behavioral richness, behavior-archive
   selection produces reward populations with measurably higher behavioral
   diversity than Pareto-over-task-instances, at matched compute budget.
2. The downstream utility of these populations differs: behavior-archive
   selection produces policy sets that are more robust to dynamics
   perturbations than Pareto-selected sets.
3. The size of these effects depends on task structure (richer-basin tasks
   show larger effects).

**Does not claim:**

- That behavior archives are universally better (they have higher descriptor-
  engineering cost; we discuss this)
- Generalization to high-DoF manipulation (out of scope; future work)
- Formal results about reward equivalence classes or shaping invariance
  (this is empirical pilot work, not theory)
- That LLM-generated reward diversity is "real" diversity (the harsh-
  reviewer-killed framing; we sidestep it)

**Pre-registered failure modes** (these are publishable outcomes, not project
failures):

- F1: Behavior archives and Pareto produce statistically indistinguishable
  populations after PPO. Implication: PPO's inductive biases dominate
  selection mechanism. Workshop-publishable as a negative result with
  diagnostic value.
- F2: Behavior archives win on diversity but lose on task performance.
  Implication: there's a cost to diversity that the paper should characterize.
- F3: Results are task-structure-dependent in unexpected ways. Implication:
  publishable as "when does each mechanism win" analysis.

If results match the predicted "behavior archives win on diversity AND
downstream robustness with minimal performance cost," paper is strong. If
they match F1-F3, paper is still publishable with appropriate framing.

---

## Tasks (Mac CPU friendly)

Three Gymnasium environments, chosen to span behavioral richness:

| Task | ID | Behavioral richness | Train time / seed |
|---|---|---|---|
| Pendulum | Pendulum-v1 | Low (one strategy) | ~1 min |
| MountainCar | MountainCarContinuous-v0 | Medium (momentum vs. push) | ~3 min |
| BipedalWalker | BipedalWalker-v3 | High (multiple gaits) | ~12 min |

The behavioral-richness gradient is the experimental design. Predicted: the
two selection mechanisms diverge most on BipedalWalker, least on Pendulum.

**Do not** add HalfCheetah / Hopper unless Mac MuJoCo install verified
working in the setup phase. Track its install time honestly.

---

## What Claude Code is responsible for

- Environment setup (Mac CPU, Gymnasium, stable-baselines3, GEPA fork)
- A GEPA fork adapted for reward search (treating reward code as the
  "prompt" being optimized)
- A behavior-archive variant of the same loop (swap Pareto selection
  for archive insertion)
- Hand-designed behavioral descriptors per task (we'll define these below)
- PPO training pipeline using stable-baselines3 on CPU
- Diversity measurements in both reward space and policy space
- Perturbation-robustness evaluation
- Statistical tests with proper seed handling
- Figures and tables for the paper
- `results.md` (technical report) and `critique.md` (self-critique)

**Claude Code is not responsible for** writing the paper itself. The human
author writes the paper. Claude Code produces the artifacts the author writes
from.

---

## Repository layout

```
gepa_archive_ablation/
├── README.md
├── protocol.md                       # this file, copied in
├── env/
│   ├── setup.sh                      # Mac CPU installer
│   ├── requirements.txt
│   └── verify_env.py                 # imports + smoke test
├── reward_gen/
│   ├── llm_client.py                 # OpenAI wrapper
│   ├── prompts/
│   │   ├── pendulum.txt
│   │   ├── mountaincar.txt
│   │   └── bipedalwalker.txt
│   ├── reward_template.py            # signature + safety wrapper
│   ├── gepa_pareto.py                # GEPA with task-instance Pareto
│   ├── gepa_archive.py               # GEPA with behavior archive
│   └── pool/                         # generated rewards, by task and method
├── training/
│   ├── train_ppo.py                  # single PPO run under given reward
│   ├── ppo_configs.py                # per-task hyperparameters (frozen)
│   └── checkpoints/
├── rollouts/                         # collected rollouts per policy
├── descriptors/
│   ├── pendulum_descriptor.py
│   ├── mountaincar_descriptor.py
│   └── bipedalwalker_descriptor.py
├── diversity/
│   ├── reward_distance.py            # code edit + structural metrics
│   ├── policy_distance.py            # trajectory + occupancy metrics
│   └── archive_metrics.py            # MAP-Elites archive coverage etc.
├── robustness/
│   ├── perturbations.py              # mass, friction, gravity perturbations
│   └── evaluate_under_perturb.py
├── analysis/
│   ├── headline_comparison.py        # the central table
│   ├── ablations.py
│   ├── statistical_tests.py
│   └── figures.py
├── figures/                          # output
├── tables/                           # output
├── logs/
├── results.md                        # technical report
└── critique.md                       # self-critique
```

---

## Setup phase

### Dependencies (verify they install cleanly on Mac M-series)

```
gymnasium[classic-control,box2d]>=0.29.0
stable-baselines3>=2.2.0
torch>=2.0.0  # CPU-only is fine; do not pull cuda wheels
numpy
scipy
scikit-learn
pandas
matplotlib
seaborn
openai>=1.0.0
ast-comments  # for reward code parsing
zss            # tree edit distance for AST comparison
```

GEPA itself: clone https://github.com/gepa-ai/gepa and `pip install -e .`.
The library is small and pure-Python; should install fine on Mac.

Box2D (for BipedalWalker) sometimes has install issues on M-series. If
`pip install gymnasium[box2d]` fails, document the failure and try
`pip install swig && pip install gymnasium[box2d]`. If that also fails,
**reduce scope to Pendulum + MountainCar only** and document the reduction
in `critique.md`.

### Verification (`env/verify_env.py`)

Must:
1. Import all libraries above
2. Instantiate each task and run 100 random-action steps
3. Make a successful OpenAI API call (use `gpt-4o-mini` to save cost during
   testing)
4. Train PPO on Pendulum for 10,000 steps as a smoke test (~10 sec on Mac)
5. Print Python version, torch version, platform, CPU count
6. Exit nonzero with a clear error if any step fails

If verification fails, **stop and report to the human author**. Do not try to
proceed with broken dependencies.

---

## Phase 1: Reward generation

### Common prompt scaffolding

For each task, the LLM receives:
- The environment's observation and action spaces
- The default hand-engineered reward (if available) as a starting point
- A docstring describing what successful behavior looks like
- Instructions to output a Python function `compute_reward(obs, action, next_obs, done)`
- A safety instruction: returned value must be a finite scalar

### Two selection mechanisms

Both use the same LLM (gpt-4o-mini for cost, gpt-4o for final runs) and
identical generation prompts. They differ only in selection:

**Variant A: GEPA-Pareto (baseline)**
- Implement standard GEPA loop adapted for reward search
- Treat each "task instance" as a different initial state of the environment
  (sample 20 initial states; these are the Pareto axes)
- Score each candidate reward by training a short PPO (50K steps) under it,
  then evaluating return from each of the 20 initial states
- Use GEPA's standard Pareto frontier selection over these 20-dimensional
  score vectors
- Use GEPA's reflective mutation (LLM reads training curve + final scores
  and proposes a new reward)

**Variant B: Behavior-Archive (proposed)**
- Identical generation, identical scoring, identical reflective mutation
- *Difference*: instead of Pareto-by-task-instance, maintain a MAP-Elites
  archive indexed by behavioral descriptor (defined per task below)
- Each candidate reward → trained short policy → policy behavior → archive cell
- Within each cell, keep only the best-performing reward
- New candidates inserted by sampling a parent uniformly from the archive

### Budget

Per task, per variant:
- 50 LLM calls total (matches Eureka's typical budget)
- 50 short PPO trainings (50K steps each; ~1-3 min per training depending
  on task)
- Total per task per variant: ~1-3 hours wall-clock on Mac

Three tasks × two variants × ~2 hours = ~12 hours of generation phase.

### Pre-flight checks

Before running Phase 1, on Pendulum only:
- Run Variant A for 10 iterations, confirm Pareto frontier updates sensibly
- Run Variant B for 10 iterations, confirm archive cells get filled
- Confirm both produce *some* nonzero-return rewards
- If either fails to produce usable rewards in 10 iterations, **stop and
  report** — there's likely a bug in the adaptation of GEPA

---

## Hand-designed behavioral descriptors

For each task, define a 2-3 dimensional descriptor space. These are critical;
arbitrary descriptors will produce arbitrary archive structure.

### Pendulum-v1

Descriptor: (mean angular velocity during episode, fraction of episode within
±0.2 rad of upright)
- Captures "did it swing aggressively or gently" × "did it stay balanced"
- 2D, each axis binned into 5 cells → 25 cells total

### MountainCarContinuous-v0

Descriptor: (max position reached before first success, mean absolute action)
- Captures "did it build momentum gradually or push hard immediately"
- 2D, 5×5 = 25 cells

### BipedalWalker-v3

Descriptor: (gait frequency from hip joint, mean forward velocity, mean
vertical body oscillation)
- Captures classic locomotion behavioral axes
- 3D, 4×4×4 = 64 cells

These descriptors are *deliberately simple* and follow established QD-
robotics convention (Cully et al. 2015 style for locomotion). The paper
should not claim novel descriptor design; descriptors are a controlled
experimental variable.

---

## Phase 2: Full PPO training

After Phase 1, you have ~50 candidate rewards per (task, variant) combination.
For Phase 2, take the top 20 rewards from each pool by Pareto rank or archive
performance, and train *full* policies (not the 50K-step proxies from Phase 1):

- Pendulum: 200K steps
- MountainCar: 500K steps
- BipedalWalker: 2M steps

5 seeds per reward. So per task per variant: 20 × 5 = 100 PPO trainings.

**Total Phase 2 PPO runs:** 3 tasks × 2 variants × 100 = 600 runs.

**Wall-clock estimate on Mac M-series:**
- Pendulum: 100 × 2 min = 200 min per variant = ~7 hours total
- MountainCar: 100 × 5 min = 500 min per variant = ~17 hours total
- BipedalWalker: 100 × 20 min = 2000 min per variant = ~67 hours total

**Total: ~90 hours of Phase 2 compute.** This is the dominant cost.

### Reducing if needed

If wall-clock proves too long after Pendulum/MountainCar complete:
- Drop to 3 seeds (was 5) on BipedalWalker → halves cost
- Drop to top-10 rewards per pool (was top-20) → halves cost
- If still too slow, drop BipedalWalker entirely; rely on the two simpler
  tasks for the main claim, with BipedalWalker as "preliminary results"

Document any scope reductions in `critique.md`.

---

## Phase 3: Diversity measurement

### Reward-space distance

For each pair of rewards within a pool, compute:
1. **AST tree edit distance** (using `zss`) on parsed reward code
2. **Embedding distance** (mean-pooled sentence-transformer embedding of the
   code as text)
3. **Behavioral distance of the reward** (run both rewards on a fixed set of
   1000 random trajectories sampled from a random policy; compute correlation
   of the two reward vectors). This is more meaningful than syntactic
   distance because it measures *what the reward actually rewards*.

Report all three. They will not agree perfectly; that's the point.

### Policy-space distance

For each pair of trained policies within a pool, compute:
1. **Trajectory DTW distance** over 100 rollouts from fixed initial states
2. **Action-distribution KL divergence** at 1000 fixed states
3. **Behavioral-descriptor distance** (the same descriptor used by the
   archive variant — but applied to both variants' policies for fair
   comparison)
4. **State-occupancy total variation** estimated from rollouts

Report all four. Triangulate; do not pick one favorable metric.

### Archive coverage metrics

For the behavior-archive variant:
- Cells filled / total cells
- Mean quality within filled cells
- QD score (sum of normalized fitnesses across filled cells)

These are standard QD reporting metrics.

---

## Phase 4: Downstream robustness

The real test of "useful diversity" is whether the diverse population
contains policies that succeed under conditions where the single-best
policy fails.

### Perturbation protocol

For each task, define a perturbation set:
- **Pendulum**: ±30% gravity, ±20% pole mass, ±20% pole length (3 axes × 3
  levels = 9 perturbed environments + nominal)
- **MountainCar**: ±30% gravity, ±30% car power, ±20% hill steepness
- **BipedalWalker**: ±20% gravity, ±30% friction, random terrain seed
  variation

For each perturbed environment:
1. Run each policy from the pool for 50 evaluation episodes
2. Compute success rate (task-defined success criterion)
3. The pool's "perturbation robustness score" = max over policies of success
   rate, summed across perturbations

### The headline comparison

For each task:
- GEPA-Pareto pool robustness vs. Behavior-Archive pool robustness
- Same compute budget for both
- Same evaluation protocol
- Report with 95% CI from bootstrap resampling

This is the central table of the paper.

---

## Phase 5: Statistical analysis

### Required tests

For all headline comparisons:
- Bootstrap 10,000 resamples of the seed dimension
- Report mean and 95% CI
- Compute Mann-Whitney U test for non-parametric comparison
- Apply Bonferroni or Holm correction for multiple comparisons (you'll be
  doing 3 tasks × multiple metrics)

### Predicted seed-noise issue

Per the harsh reviewer's Core Problem 8: optimization noise across seeds may
dominate reward differences. To diagnose:
- Compute the *same* analysis but using 5 different seeds of a *single*
  reward function. This is the noise floor.
- If pool-level diversity (between rewards) is not larger than seed-level
  diversity (within reward), your selection mechanism is selecting noise.
- Report this explicitly. It is honest and strengthens the paper.

### What to do if seed noise dominates

If the noise floor exceeds the mechanism effect:
- Increase seed count where feasible
- Reframe paper as "we observe that PPO seed variance dominates reward
  variance in classic control" — this is itself a workshop-publishable
  finding
- Do not hide this result

---

## Phase 6: Analysis and writeup

### `results.md` (technical report — Claude Code writes this)

Required sections:
1. **Summary of findings** — 1 paragraph, claims-only, no fluff
2. **Headline numbers** — the central table comparing both variants on all
   three tasks, with CIs
3. **Per-task analysis** — what happened on Pendulum, MountainCar,
   BipedalWalker; differences explained
4. **Diversity metric agreement** — do the multiple diversity metrics agree?
   If not, why?
5. **Seed noise floor** — explicit comparison; mechanism vs. noise
6. **Failure modes encountered** — what didn't work, what was skipped, what
   was scope-reduced
7. **Pre-registered hypotheses outcomes** — did predicted outcomes occur?

Write this in plain prose. Do not write paper-style hype. The human author
writes the paper from this report.

### `critique.md` (self-critique — Claude Code writes this)

Required sections:
1. **Threats to validity** — what could make the headline claim wrong?
2. **What we did not measure** — limitations
3. **What we should have done with more compute** — full multi-seed sweeps,
   more tasks, longer training
4. **Possible reviewer objections** — anticipated and responded to, with
   honest acknowledgement when an objection cannot be fully addressed
5. **Comparison to prior work that we didn't run** — Eureka, DrEureka,
   RoboMoRe; what would the proper comparison look like and why we didn't do it

Be harsh in `critique.md`. The human author needs to know what reviewers will
say.

### Figures Claude Code must produce

1. **Figure 1**: Headline comparison bar chart — pool robustness for both
   variants across three tasks, with error bars
2. **Figure 2**: Diversity metrics scatter — reward-space distance vs.
   policy-space distance, colored by variant
3. **Figure 3**: Archive coverage plot for behavior-archive variant
4. **Figure 4**: Seed-noise floor comparison — noise floor vs. mechanism
   effect, per task
5. **Figure 5**: Per-task perturbation breakdowns

All figures: clean, publication-quality, saved as both PNG (preview) and PDF
(submission). Use seaborn defaults; do not overstyle.

---

## What to do if things go wrong

### Common failure modes and responses

1. **GEPA adaptation doesn't work** (reflective mutation degenerates,
   archive doesn't fill, etc.): document, scope-reduce to "we report
   preliminary results on the adaptation we got working" if partial. If
   neither variant works, **stop and report to human**.

2. **PPO doesn't converge on a task** for many candidate rewards: this is
   informative. Report the failure rate. It may be the finding ("most
   LLM-generated rewards fail to produce trained policies on task X").

3. **BipedalWalker too slow:** drop to MountainCar and Pendulum only.
   Document.

4. **Box2D install fails on Mac M-series:** drop BipedalWalker, document
   in `critique.md`, proceed with Pendulum and MountainCar.

5. **OpenAI API costs growing too fast:** stop, report current spend, ask
   human to decide whether to continue. Cap at $50 total for the project
   unless human explicitly authorizes more.

6. **Results look uninterpretable:** stop and report. Do not torture data
   until something significant appears. Negative or null results are
   acceptable.

---

## Honesty requirements

Claude Code must:

- **Log all failed runs, scope reductions, and unexpected behaviors** in
  `logs/`. The human author needs this to write an honest paper.
- **Not cherry-pick seeds, metrics, or hyperparameters** to make results
  look better. Report what was run, not what worked best.
- **Document compute cost (wall-clock, API spend)** explicitly. Workshop
  reviewers increasingly ask for this.
- **State explicitly in `results.md` what each claim is and isn't supported
  by**. Two seeds of Pendulum is not "evidence on locomotion tasks."

If at any point Claude Code is uncertain whether to proceed (ambiguous
results, partial failures, budget concerns), stop and ask the human author.
Do not silently make scope decisions that affect what the paper can claim.

---

## Estimated total wall-clock on a MacBook M-series

- Setup + verification: ~2 hours
- Reward generation (Phase 1): ~12 hours
- Full PPO training (Phase 2): ~90 hours (the dominant cost)
- Diversity measurement (Phase 3): ~2 hours
- Robustness evaluation (Phase 4): ~10 hours (extra rollouts under
  perturbations)
- Analysis and figures: ~2 hours

**Total: ~120 hours = ~5-6 days of continuous compute.**

In practice, MacBooks throttle under sustained CPU load. Realistic schedule:
**2-3 weeks of part-time work** running batches overnight.

If the human author needs faster turnaround, the only honest answer is to
get GPU access. CPU-only at this scale is the bottleneck, not the algorithm.

---

## What this paper does NOT do (for reviewers' future reference)

Items the human author should explicitly list as out-of-scope, anticipating
reviewer questions:

- No comparison to Eureka or DrEureka as separate methods (we use them as
  conceptual references but do not benchmark against their exact
  implementations)
- No theoretical results about reward equivalence classes
- No claims about sim-to-real
- No claims about high-DoF manipulation
- No claims that hand-designed descriptors are universally optimal — we
  acknowledge the descriptor-engineering burden and discuss it

Naming these limits upfront in the paper makes reviewers less likely to
demand them in review.