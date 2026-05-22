# Research Chat Log — Curiosity-as-Pareto-Axis × EUREKA × Tactile

> Living transcript of the design conversation. Newest exchanges at the bottom.
> Started 2026-05-21. Paper working title: *Curiosity as Pareto Frontier: GEPA Reward Tuning for Sparse Reward Learning.*

---

## 1. Project nutshell — first framing

**User:**
> we know that LLM can control robot policy by outputting and tuning reward evolutionarily (EUREKA paper) →
> and EUREKA didn't have a diverse candidate selection →
> Therefore we can apply the idea of GEPA's Pareto frontier pool into Eureka to improve it on deceptive task and non-convex landscape →
> and Additionally, we can add curiosity as its own Pareto axis in the GEPA process such that a weaker overall candidate (in our situation, a REWARD FUNCTION that performs worse on all example scenarios) that would've gotten dominated and kicked out in regular GEPA would get to stay.
>
> The simple task remains to tune it on a sparse reward game program and show GEPA + EUREKA + CURIOSITY beats plain GEPA + EUREKA. (Eureka is involved because we borrow the idea of REWARD FUNCTION being the evolutionary candidate)
>
> further stretch goal: change the sparse reward game program into a sim tactile robotic task → this will allow us to integrate tactile curiosity signal — which is another area of contribution.
>
> current risk: too many areas of contribution — narrow down in one
> 1. valid curiosity as pareto frontier?
> 2. tactile signal for curiosity?
> 3. gepa + eureka (altho this is too easy)
>
> no time to do sim-to-real transfer before CORL 2026.

**Claude:**

**Thesis chain**
1. EUREKA → LLMs can evolve reward functions for robot policies.
2. Gap → EUREKA's selection lacks diversity, so it gets stuck on deceptive / non-convex reward landscapes.
3. Fix → graft GEPA's Pareto-frontier pool onto EUREKA's reward-function evolution.
4. Novelty → add **curiosity as its own Pareto axis** so a reward candidate that loses on every task scenario but drives exploratory behavior survives selection instead of being dominated out.

**Empirical claim:** on a sparse-reward game, GEPA+EUREKA+Curiosity > GEPA+EUREKA.
**Stretch:** swap the game for a sim tactile task → opens a second contribution (tactile-derived curiosity signal).

**Risks flagged:** three candidate contributions (curiosity-as-Pareto-axis, tactile curiosity, GEPA+EUREKA), and #3 alone is too thin; no sim-to-real bandwidth before CORL 2026.

**Read on the narrowing question:**

Anchor the paper on **#1 (curiosity as a Pareto axis)** and treat tactile curiosity as future work, not a co-headline. #1 is the actual methodological claim — it stands on the sparse-reward game alone, doesn't depend on tactile hardware, and fits the CORL timeline. If #1 lands, tactile curiosity is a natural follow-up paper; if you try to ship both, neither gets defended properly. The main tradeoff: a pure-#1 paper is "yet another exploration bonus" framed as Pareto selection — you'll need a sharp story about *why the Pareto framing matters* (e.g., dominated-but-explorative candidates that vanilla EUREKA culls) to avoid that reading.

> *(Memory saved: project_research_thesis.md)*

---

## 2. Benchmark recommendation

**User:**
> what should we trained this on as a benchmark

**Claude:**

**Lead with MiniGrid (KeyCorridor / ObstructedMaze / MultiRoom) + AntMaze-medium as the headline.** Both are canonical, cheap to run, and each isolates exactly the property the thesis needs.

**1. MiniGrid — the curiosity proving ground.** Sparse-reward, deceptive (locked doors, hidden keys), symbolic state that an LLM can trivially write reward code over (EUREKA-style). It's the standard benchmark in every curiosity paper since ICM/RND, so reviewers immediately understand what "curiosity helped" means. Iteration is seconds-per-rollout, which matters when GEPA's Pareto pool means you're training many policies per generation. This is where you'd actually *demonstrate* that the curiosity Pareto axis rescues dominated candidates.

**2. AntMaze (D4RL, medium or large)** — gives the "robotic" flavor CORL wants. Sparse reward, deceptive geometry (must go around walls), continuous control. LLM can write rewards over `(x, y, vel, goal)`. Slower than MiniGrid but still tractable.

**3. (Optional) A sparse-ified subset of EUREKA's Isaac Gym tasks** — strongest narrative ("same benchmark as EUREKA, plus curiosity") but you'd have to strip the dense reward signals yourself, and Isaac Gym setup is the kind of yak-shave that eats CORL timelines.

**Main tradeoff:** MiniGrid alone risks the "not robotic enough for CORL" reaction. AntMaze alone risks "the curiosity benefit is too subtle to see in N seeds." Doing both gives a clean two-panel story: *MiniGrid shows the mechanism, AntMaze shows it transfers to continuous control.*

Avoid for primary experiments: anything pixel-based (Atari/Procgen) — EUREKA's premise is the LLM reading symbolic state to write reward code, and pixels break that.

---

## 3. Pluggable framework question + EUREKA/field research

**User:**
> should we implement this as a pluggable reward generation framework for any task? aka it will tune reward for any upstream robotic architecture PPO, etc. in that case what need to be exposed and connected to the framework? im thinking #1 — a way for framework to manipulate and give reward, #2 a way for framework to reason about this envioronment? how to make it pluggable while not losing original Eureka paper's proposal on LLM reasoning for physical landscape? how is this done in Eureka (pull actual research) and how is this currently done in field of robotic? do people tend to have framework for this kind of thing?

**Claude (after pulling EUREKA paper, Text2Reward, CARD, VIRAL):**

### EUREKA's actual interface (arXiv 2310.12931)
- **LLM context:** raw env source code — specifically the `compute_observations()` function, extracted automatically — plus a natural-language task description. **No structured observation/action spec is passed**; the LLM infers available variables by reading the code.
- **LLM output:** a `@torch.jit.script` Python function `compute_reward(...) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]`. Returns `(scalar reward tensor, dict of named reward components)`. The function reads `self.*` attributes of the env.
- **Reward reflection:** per-component reward trajectories snapshotted at intermediate policy checkpoints, plus success rate, episode length, max/mean/min per component. Fed back as text into the next LLM generation.
- **Fitness function (selection):** a **separate task-success metric `F`**, *not* the LLM-generated reward. Load-bearing detail — selection uses ground truth, generation uses LLM output. **Curiosity-as-Pareto-axis hooks in here.**
- **RL stack:** PPO from `rl_games` on Isaac Gym, 5 seeds per final candidate, 1 seed for intermediate selection.
- **Search budget:** K=16 samples/iteration, N=5 iterations.

### Text2Reward (Xie et al. 2024)
- Env representation = hand-written **"compact Pythonic class spec"** (typed attributes + docstrings), not raw source. More portable but loses some richness.
- Reward signature `compute_dense_reward(self, action) -> float`. Helper APIs: numpy/scipy/transforms3d.
- Targets MetaWorld + ManiSkill2 + MuJoCo, but each requires a hand-crafted env abstraction.

### CARD (LLM-Driven Reward Design via Dynamic Feedback, 2410.14660)
- Same Pythonic-spec representation as Text2Reward. Adds a rule-based Evaluator to cut LLM queries (1/iter vs EUREKA's 16). Still tied to MetaWorld + ManiSkill2 abstractions.

### VIRAL (2505.22092)
- Vision-grounded; works from raw observations via VLM. Maximum portability but drops EUREKA's premise of LLM reading code/state semantics. Wrong direction for this project.

### Verdict: does a standard pluggable framework exist?
**No.** Several attempts (CARD, VIRAL, LLM2Rewards ROS Evolution) exist but none have become a dominant interface like `gymnasium.Env`. EUREKA itself is hand-wired per Isaac Gym task. The field has no `RewardDesigner` standard, which means a thin, well-designed harness is both feasible and a credible *secondary* contribution.

### Recommendation: thin harness, not a general framework

Build the interface so two envs (MiniGrid + AntMaze) can run, but don't pre-generalize for "any Gymnasium env." A general framework is engineering work that does not strengthen the curiosity-as-Pareto-axis claim. The polished framework is the natural follow-up paper if the curiosity-as-Pareto-axis claim lands.

### Interface contracts the harness should expose

1. **EnvAdapter** — what the LLM sees. Returns `observation_source` (code string), `task_description`, `reward_signature`, `helper_apis`. ~30–80 LoC per env. Follows EUREKA's code-as-context, not Text2Reward's Pythonic-spec.
2. **RewardInjector** — accepts LLM-emitted Python string, validates (parse + sandboxed dry-run), injects into a wrapper env.
3. **TrainerAdapter** — input: env + compute_reward + seed + step budget. Output: `CandidateResult(F, C, per_component_traj, training_curve, success_rate, episode_length)`. Start with Stable-Baselines3 PPO.
4. **ParetoSelector** — maintains population over `(F, C)` axes. Ablation flag `--axes task_only` vs `--axes task+curiosity`.

*(Plan written to `/Users/lala/.claude/plans/vast-splashing-moth.md`)*

---

## 4. Design forks resolved

User locked in (via AskUserQuestion):
- **Pluggability:** Thin harness, 2 envs (MiniGrid + AntMaze). Per-env adapter ~50 LoC.
- **Env representation:** Env source code, EUREKA-style. Richest physical reasoning; matches EUREKA's premise.
- **Curiosity signal:** held off for clarification (see next section).

---

## 5. Curiosity signals explained — state-visitation entropy vs RND

**User:**
> please explain the two type of curiosity signal

**Claude:**

### State-visitation entropy

**What it measures:** how *spread out* a policy's state visits are.

**How it works:** bin the state space → roll out policy N episodes → count bin visits → normalize → Shannon entropy `H = -Σ p(s) log p(s)`. Higher H = policy spreads out; lower H = policy camps in a few states.

**As a Pareto axis:** for each candidate reward `R`, train policy `π_R`, report `(F, C=H(visitation of π_R))`. Reward with low task success but high-entropy policy survives the front.

**Pros:** essentially free (counts in a dict), interpretable, no NN to train. Trivial on MiniGrid.
**Cons:** continuous states need binning choices reviewers will pick at. Says nothing about whether states are *novel* — uniform camping in a small region has high entropy.

### RND (Random Network Distillation)

**What it measures:** how *novel* a policy's visited states are, relative to what a small network has learned to predict.

**How it works:** `f_target` (random, frozen) and `f_pred` (trained to mimic f_target on visited states). Novelty = ‖f_target(s) − f_pred(s)‖². High on states f_pred hasn't seen enough; low on familiar.

**As a Pareto axis:** `C = mean RND error along π_R's rollouts`.

**Pros:** no binning — continuous/high-dim/pixel friendly. Standard exploration-literature recognition.
**Cons:** extra NN per candidate. Predictor-net hyperparameters affect signal. Needs normalization.

### Conceptual difference

- **Entropy = spread.** Distribution-level.
- **RND = novelty.** State-level, aggregated.

A policy that loops uniformly through 10 boring states scores high entropy / low RND.

### Pareto-axis framing

Lean **entropy** for the headline claim:
- Entropy is a property of the policy R produces, not a secondary reward channel. Defends as "independent objective dimension" rather than "intrinsic reward in disguise."
- With RND, reviewers can argue "you're just adding RND-as-bonus in a costume — why not sum RND into reward and use vanilla EUREKA?"

Add **RND as a robustness ablation** in appendix.

---

## 6. ICM vs RND — drawbacks

**User:**
> so basically the f_target - f_predict idea kinda embedd the idea of an encoder in 2017 ICM paper into f_predict? what are the drawbacks?

**Claude:**

Half right — both have something encoder-shaped, but the resemblance is shallow.

### ICM (Pathak 2017) — three networks, dynamics-aware
- **Encoder φ** — learns state → features.
- **Inverse model** — predicts `a_t` from `(φ(s_t), φ(s_{t+1}))`. Load-bearing: forces φ to encode only *action-controllable* features (filters noisy-TV).
- **Forward model** — predicts `φ(s_{t+1})` from `(φ(s_t), a_t)`. Curiosity = ‖predicted − actual‖² in φ-space.

Novelty signal = **transition prediction error in a learned, action-relevant feature space.**

### RND (Burda 2018) — two networks, no dynamics
- f_target (random, frozen), f_pred (trained to mimic).
- Novelty = ‖f_target(s) − f_pred(s)‖².

Novelty signal = **how much exposure the predictor has had on this state**, against an arbitrary random projection.

RND's "encoder" is random — no semantics. The contribution of RND was throwing away the dynamics model entirely and showing it still works.

### RND drawbacks

1. **No action awareness.** Can't distinguish states the agent caused from states env produced.
2. **Random target is semantically arbitrary.** Similar states can be far; different states can be close. Empirically works, no principled reason.
3. **Predictor capacity is finicky.** Too big → saturates; too small → never learns. Doesn't transfer across envs.
4. **Saturation over training.** As policy revisits, f_pred converges → novelty collapses. For Pareto axis: *timing* of C measurement matters; re-init per candidate or measure early.
5. **No transition novelty.** Rewards rare states, not surprising consequences. If deceptive landscape hinges on transitions (a key unlocking reachability), RND undervalues it vs ICM.
6. **Noisy-TV: a wash.** RND > forward-dynamics on noisy-TV (predictor learns marginal of random function over noise). ICM with inverse-dynamics also handles it.

### Where this lands

- **Sparse-reward navigation (MiniGrid, AntMaze):** RND > ICM. Question is "did agent reach unfamiliar parts of state space" — exactly RND.
- **Tactile stretch goal:** ICM > RND. Tactile curiosity = contact events being surprising = transition signal in learned feature space = ICM-shaped.

Picking RND now means rebuilding for tactile later.

---

## 7. CORL venue fit — harsh

**User:**
> would this simple MiniGrid AntMaze set up even be accepted to a conference like CORL 2026

**Claude:**

Honestly: **MiniGrid + AntMaze alone is a stretch for CORL.**

### The CORL bar

CORL = Conference on Robot Learning. Reviewers heavily weight physical embodiment, robotics relevance, real-robot or robot-aligned sim.

- **MiniGrid:** not a robot. Right benchmark for NeurIPS/ICLR/ICML exploration papers. CORL reviewers will mark down — even as "mechanism demonstration" appendix.
- **AntMaze:** legitimately a robot (simulated quadruped). Counts. But single sim-robot env, locomotion-only, no contact-rich manipulation = thin as the *only* robotic evidence.

Acceptance odds at CORL 2026 with MiniGrid + AntMaze: **low-to-marginal**. Better venue: **ICLR or NeurIPS** (exploration / LLM-for-RL track).

### What would actually be CORL-grade

- **MetaWorld sparse variants** (door, drawer, button, sweep) and/or **ManiSkill2** (PickCube, TurnFaucet, PushChair). Strongest CORL fit; direct lineage from EUREKA-space.
- **Isaac Gym subset from EUREKA's 29 tasks** (sparse-ified). Strongest "we extend EUREKA" pitch. Heavier setup.
- **Robosuite manipulation** with sparse rewards.

### Timeline reality

Today: 2026-05-21. CORL 2026 deadline historically falls in June → **2-6 weeks out**.

Three paths:
- **Path A (CORL push):** pivot to MetaWorld + AntMaze, race the deadline.
- **Path B (better venue match):** keep MiniGrid + AntMaze, target ICLR 2027 — exploration/LLM-for-RL fit, 4-5 months to add manipulation env.
- **Path C (workshop):** CORL workshop (lower bar) + main-venue ML conference for full paper.

Recommendation: **Path A or B, not C.** Workshops don't give the CV line.

---

## 8. Tactile pivot — harshness + Text2Touch bombshell

**User:**
> please be harsh. explain metaworld. i would also like to clarify the vision — incorporate curiosity-infused gepa specifically onto tactile task, with curiosity signal primarily coming from touch but understandable from LLM. how is tactile task lately being trained (corl 2025) and how could we add GEPA?

**Claude (after pulling Text2Touch, TacSL, Tactile Gym 2.0, Sparsh, MetaWorld):**

### The uncomfortable finding: Text2Touch (CoRL 2025)

[Text2Touch, arXiv 2509.07445](https://arxiv.org/html/2509.07445v1) — accepted CoRL 2025, eight months ago:
- Task: multi-axis in-hand object rotation, four-fingered dexterous hand, TacTip tactile sensors, Isaac Gym sim, sim-to-real.
- Method: **EUREKA pipeline directly applied to tactile.** GPT-4 → `compute_reward` → PPO → structured feedback → repeat. **K=4, N=5 = 20 reward functions.**
- Selection: **best-of-K, not Pareto.**
- Curiosity / intrinsic motivation: **none.**
- LLM context includes **70+ env variables, many tactile** — "LLM-understandable tactile reward" is solved.

**Their explicitly listed limitations** read like the contribution slot:
- *"Low solve rate in reward discovery phase; many LLM-generated functions fail to train an agent to task completion."*
- Single-task. Best-of-K.

**Repositioned thesis:**
> Text2Touch shows LLMs can design tactile rewards but reports low solve rate because best-of-K selection kills diverse candidates. We add GEPA's Pareto frontier with tactile-derived curiosity as an axis, recovering dominated-but-explorative candidates and lifting solve rate on Text2Touch's benchmark.

Sharper than MiniGrid + AntMaze. But has costs.

### MetaWorld — explainer + why it's now wrong

- 50 tabletop manipulation tasks, 7-DOF Sawyer arm, MuJoCo.
- Tasks: reach, push, pick-place, drawer, peg insertion, door, button, sweep, shelf, basketball, window.
- Each task has **dense shaped reward + binary success metric** — sparse-reward variants trivial.
- Standard splits: ML10 / MT10 / MT50 / ML45.
- **No tactile sensors.** Dealbreaker for the new tactile-curiosity vision.

### How tactile RL is actually trained now (CoRL 2024-2025)

- **TacSL (T-RO 2025).** GPU-accelerated visuotactile sim on Isaac Gym. 200× faster than prior. Algorithm: asymmetric actor-critic distillation. **De facto tactile sim infrastructure.**
- **Tactile Gym 2.0** — CPU-bound, older, mostly superseded.
- **Sparsh (Meta 2024).** Self-supervised touch representations across DIGIT/GelSight. Used as frozen encoder.
- **VITAL / SaTA / FARM (2024-25)** — tactile-conditioned policies (diffusion, transformer). Policy-side, not reward-design.
- **Text2Touch (Sep 2025)** — LLM reward design on tactile. The predecessor.

Modal CoRL 2025 pipeline: Isaac Gym + TacSL → tactile encoder (Sparsh) → PPO/diffusion policy → sim-to-real on DIGIT/GelSight/TacTip. Multi-month effort.

### How GEPA + curiosity slots in on Text2Touch's gaps

- **Pareto pool replaces best-of-K.** Maintain Pareto front over `(F, C)`. Mutate parents drawn from front.
- **Tactile curiosity axis options:**
  - RND on tactile feature vectors (contact maps from TacSL).
  - Entropy of contact-region histograms.
  - **ICM-style forward-model error on tactile readings** — most aligned with "tactile = transition-novelty matters." Closest to how human tactile curiosity works.
  - Force-trajectory entropy (cheap).
- **LLM-understandable tactile features.** Text2Touch's 70+ variables inherited.

### Harsh timeline reality

From blank repo to CORL-quality tactile paper beating Text2Touch:
- Isaac Gym install + working PPO: **1-2 weeks**.
- TacSL install + tactile env: **1-2 weeks**.
- Reproduce Text2Touch baseline: **2-4 weeks**.
- Implement Pareto + tactile curiosity: **1-2 weeks**.
- Experiments at Text2Touch scale (5 rewards × 8B steps × 5 reps): **2-4 weeks GPU**.
- Writing: **1-2 weeks**.
- **Total: 8-16 weeks. Available: 2-4.**

**CORL 2026 is not happening.**

### Realistic venue pivot

- **RSS 2027** (deadline ~Jan-Feb 2027, ~8 months): best methodological-novelty fit.
- **ICRA 2027** (deadline ~Sept 2026, ~4 months): tight but doable. Manipulation-friendly.
- **CORL 2027** (deadline ~June 2027, ~13 months): luxury budget, includes sim-to-real.
- **CORL 2026 workshop**: CV line + community feedback, then full paper to RSS/ICRA.

### Harsh verdict

The "novel" part of EUREKA+tactile was published 8 months ago. Remaining novelty (Pareto + tactile curiosity) is real but reduced to a *delta over Text2Touch*. CORL 2026 from blank repo with no Isaac Gym / TacSL experience and a contribution that requires beating a CoRL 2025 paper on its own benchmark = not realistic. **RSS 2027 or ICRA 2027 is the honest goal.** Use next 6 weeks to install Isaac Gym + TacSL, reproduce Text2Touch baseline, get one Pareto-pool experiment running — strong workshop submission and foundation for a full paper.

---

## Sources collected so far

- [Text2Touch (CoRL 2025)](https://arxiv.org/html/2509.07445v1)
- [TacSL (T-RO 2025)](https://arxiv.org/abs/2408.06506)
- [EUREKA (arXiv 2310.12931)](https://arxiv.org/abs/2310.12931)
- [Text2Reward (ICLR 2024)](https://arxiv.org/abs/2309.11489)
- [CARD: LLM-Driven Reward Design via Dynamic Feedback](https://arxiv.org/html/2410.14660v1)
- [VIRAL: Vision-grounded Integration for Reward design](https://arxiv.org/pdf/2505.22092)
- [Meta-World benchmark](https://ar5iv.labs.arxiv.org/html/1910.10897)
- [Sparsh: self-supervised tactile representations](https://arxiv.org/html/2410.24090v1)
- [CoRL 2025 papers index](https://github.com/smallfryy/corl-2025-papers)
- [Tactile Intrinsic Motivation (2021)](https://arxiv.org/pdf/2102.11051)

---

## 9. Export convention set

**User:**
> please export this current convo to MD and add to it as we go

**Claude:** Wrote this file (`ideas/research_chat.md`) with sections 1–8 above and saved a memory feedback note (`feedback-chat-log`) so every substantive future exchange in this project is appended here as `## N. <topic>`. Trivial back-and-forth is not logged.

---

## 10. Scope reset → workshop-track classic-control ablation

**User:**
> please - use the instruction in iter1.md. produce all your result in iter1 folder

This is a *third* scope shift. Sections 1–6 framed sparse-reward exploration with curiosity-as-Pareto-axis. Section 8 pivoted to tactile manipulation vs Text2Touch. Section 10 (this section) hands me a new protocol — `iter1.md` — that targets a CoRL 2026 *workshop* with a Mac-CPU-only ablation of **GEPA-Pareto vs MAP-Elites Behavior-Archive** on Pendulum + MountainCar + BipedalWalker. Curiosity drops out as the headline; the contribution becomes "which diversity mechanism (task-instance Pareto vs behavior archive) is better for LLM reward search?".

The protocol estimates ~120 hours of compute (~2–3 weeks part-time) and explicitly enumerates Claude Code's vs the human author's responsibilities. Claude Code is responsible for the artifacts; the human writes the paper.

**What I built this turn (scaffold only — no experiments):**

```
iter1/
├── README.md
├── protocol.md           # copy of iter1.md
├── env/
│   ├── setup.sh          # Mac M-series CPU installer; handles swig+Box2D
│   ├── requirements.txt
│   └── verify_env.py     # imports + env smoke + OpenAI call + 10k PPO smoke
├── reward_gen/
│   ├── llm_client.py     # minimal OpenAI wrapper w/ token+cost tracking
│   ├── reward_template.py# AST-validated compile + safe_call clamp
│   ├── _loop_common.py   # shared: TaskSpec, RewardOverrideEnv, short-PPO, mutation prompt, persistence
│   ├── prompts/{pendulum,mountaincar,bipedalwalker}.txt
│   ├── gepa_pareto.py    # Variant A: non-dominated front over 20-init-state vector
│   └── gepa_archive.py   # Variant B: MAP-Elites archive
├── training/
│   ├── ppo_configs.py    # frozen Phase-2 hyperparameters per task
│   └── train_ppo.py      # full PPO run from chosen Phase 1 reward
├── descriptors/{pendulum,mountaincar,bipedalwalker}_descriptor.py
├── diversity/{reward_distance,policy_distance,archive_metrics}.py
├── robustness/{perturbations,evaluate_under_perturb}.py
├── analysis/{statistical_tests,headline_comparison,ablations,figures}.py
├── results.md            # structural placeholder
└── critique.md           # scaffold-time critique already filled (8 known issues); experiment-time TBD
```

**Knowing deviations from the protocol, documented in critique.md:**
- Did not import the real GEPA library — its prompt-optimization adapter doesn't fit reward search without monkey-patching. Reimplemented non-dominated sort + reflective mutation directly.
- MountainCar descriptor uses "max position over episode" not "max before first success" (latter is undefined for unsuccessful policies).
- AST validation blocks obvious bad imports/calls but is not adversarial-grade. Workshop scope.
- Phase 1 short-PPO is 50K steps × 1 seed per protocol; noisy but fair across variants.

**Stop checkpoint:** Claude Code is at the protocol's first explicit handoff. Per protocol §"Verification": *"If verification fails, stop and report to the human author. Do not try to proceed with broken dependencies."* Human author must now run:

```bash
bash iter1/env/setup.sh
export OPENAI_API_KEY=sk-...
python iter1/env/verify_env.py
# then if green:
python -m reward_gen.gepa_pareto --task pendulum --iters 10 --preflight
python -m reward_gen.gepa_archive --task pendulum --iters 10 --preflight
```

If preflight passes, Phase 1 can proceed. If either fails per protocol's stop conditions, report back.

---
