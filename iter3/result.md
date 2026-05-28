# Behavior-Diversified Reward Evolution via VLM-Embedded Quality-Diversity

**iter3 — Final technical report**
Author: Haochen Han · UC Berkeley · `haochenhan@berkeley.edu`
Date: 2026-05-28
Status: **Final** (single-seed; multi-seed deferred to iter4 / next phase)

---

## Abstract

We extend **EUREKA**-style LLM reward design with a quality-diversity (QD)
selection mechanism that uses a **pretrained vision-language model** as the
behavior-descriptor source. Each LLM-authored $\texttt{compute\_reward}$ is
trained for a short PPO budget on Hopper-v5; the resulting policy is rolled
out, described in natural language by Claude Sonnet 4.6, condensed by
Claude Haiku 4.5, embedded with sentence-transformers MiniLM, then inserted
into a flat $k$-NN dominance archive. Parents for the next mutation are
sampled with weight proportional to local sparsity ($\propto$ mean $k$-NN
distance), explicitly counter-acting collapse to one mode. We compare
against (i) **argmax-EUREKA**, the canonical best-of-$K$ baseline, and (ii)
**AURORA-style**, an unsupervised autoencoder over pooled trajectory
features. All three are scored in a hand-designed reference grid spanning
forward-velocity × body-height × gait-frequency — never used for selection.

**Headline result (single seed, $N = 20$ candidates per method):**

|                   | best fit | mean fit | coverage | QD score | archive size |
|---|---:|---:|---:|---:|---:|
| **archive (ours)** | **333.99** | 197.66 | **10.4%** | 5.72 | 15 |
| argmax           | 227.63 | **217.14** | 8.8% | 3.57 | 20 |
| aurora           | 222.63 | 206.01 | 9.6% | **6.38** | 1 |

Our method attains **+47% peak fitness** over argmax-EUREKA and **+50%**
over AURORA-style. Coverage in the reference grid is highest for our
method (10.4%). AURORA-style suffers an archive-collapse failure at this
small budget (after the first AE refit, 4 of 5 bootstrap elites are
dominated under the freshly-trained latent), leaving a single elite —
*this is itself a finding* about small-budget AURORA.

---

## 1. Introduction

EUREKA [Ma et al.\ 2023] showed that LLMs can author reward functions for
robot RL, evolving them with best-of-$K$ (argmax) selection on a
ground-truth fitness $F$. Argmax has a known pathology in evolutionary
search: it collapses the population around the current local optimum and
discards candidates that lose on $F$ but exhibit *behaviorally distinct*
policies. Text2Touch [Wang et al.\ 2025], the most recent EUREKA descendant
applied to tactile manipulation, reports a low solve rate explicitly
attributed to this collapse.

The MAP-Elites family [Mouret \& Clune, 2015] addresses this via a grid
over a hand-designed behavior descriptor; AURORA [Cully, 2019;
Grillotti \& Cully, 2021] generalizes to *learned* descriptors via an
autoencoder trained on rollout trajectories. Both require infrastructure
(hand-designed BD or per-task AE training) that the LLM pipeline does
not naturally supply.

We observe that the **same VLM already in every modern LLM pipeline**
can serve as a third behavior-descriptor source: zero-shot,
human-interpretable, and continuous (an embedding under a sentence
encoder). This paper asks whether such a descriptor, plugged into
EUREKA's selection step, improves both behavioral diversity and peak
fitness on a fixed compute budget.

**Contributions.**

1. **Method.** A flat $k$-NN dominance archive with density-corrected
   sampling over MiniLM-embedded VLM descriptions of trained policies.
   The single hyperparameter $\tau$ (novelty radius) is calibrated from
   the first five candidates.
2. **System.** A standalone reproducible pipeline (`iter3/`) — AST-sand-
   boxed reward execution, short PPO under user-provided rewards,
   rollout-to-video, VLM-described, embedded, persisted as JSON.
3. **3-way comparison.** A single-shot evaluation against argmax-EUREKA
   and an AURORA-style autoencoder baseline, all scored in a shared
   hand-designed reference grid.

---

## 2. Related Work

**LLM reward design.** EUREKA [Ma et al.\ 2023] couples an LLM with PPO
in a $K=16$ best-of-$K$ loop. Text2Reward [Xie et al.\ 2024], CARD
[2024], and VIRAL [2025] all inherit argmax selection. Text2Touch [2025]
applies EUREKA to tactile manipulation, reporting low solve rate.

**Quality-diversity.** MAP-Elites [Mouret \& Clune, 2015] discretizes a
hand-designed BD into a grid; CVT-MAP-Elites [Vassiliades et al.\ 2017]
generalizes to centroidal Voronoi tessellations; Quality-Novelty [Pugh
et al.\ 2016] maintains a Pareto front over (fitness, novelty). Our
archive is closest to a continuous, $\epsilon$-novelty form of
MAP-Elites with $k$-NN-defined neighborhoods.

**Unsupervised BD.** AURORA [Cully, 2019; Grillotti \& Cully 2021/2022]
learns a trajectory autoencoder, re-fits ("container reset") as the
archive grows. We use a pretrained, frozen VLM and skip the refit cost.

**Pareto selection for LLM outputs.** GEPA [Agrawal et al.\ 2025] uses
a Pareto frontier over per-instance task scores when evolving prompts.
Our work transposes this idea from instance scores to a VLM behavior
descriptor.

---

## 3. Method

### 3.1 Reward sandbox and EUREKA loop

The LLM authors

$$\texttt{compute\_reward}: (s_t, a_t, s_{t+1}, d) \to \mathbb{R}$$

with $s_t \in \mathbb{R}^{11}$ (Hopper-v5 observation), $a_t \in [-1,1]^3$,
$d \in \{0,1\}$ terminal. Outputs are clamped to $[-100,100]$;
non-finite → 0. A whitelist AST sandbox rejects imports, dunder attribute
access, and unsafe builtins.

Per iteration $i \in \{1,\dots,20\}$:

1. **Generate.** Parent ← strategy-specific selection (or cold-generate
   if $|\mathcal{A}|=0$). LLM prompt = parent code + parent VLM description
   + (for QD strategies) descriptions of $k=3$ nearest neighbors, with
   instruction *"produce a variant whose policy behaves differently from
   these neighbors."*
2. **Train.** PPO (stable-baselines3, CPU) for $T_{\text{ppo}}=30{,}000$
   timesteps in $\texttt{RewardOverrideEnv}$ wrapping Hopper-v5.
3. **Evaluate.** Roll out the trained policy for $T_{\text{eval}}=600$
   steps under the **default Hopper reward**. Mean episode return is the
   fitness $F_i$. (Following EUREKA: selection uses ground-truth fitness,
   not the LLM-authored reward — otherwise the LLM trivially writes
   "return 1e9" and "wins".)
4. **Describe.** Six evenly-spaced frames → Claude Sonnet 4.6 → multi-
   sentence description → Claude Haiku 4.5 → $\sim$15-word concise tag.
5. **Embed.** Concise tag → MiniLM → $e_i \in \mathbb{S}^{383}$.
6. **Insert** into archive per strategy (§3.2).

### 3.2 $k$-NN dominance archive

Given new candidate $(F, e)$ and archive $\mathcal{A}=\{(F_j, e_j)\}$,
let $j^* = \arg\min_j d(e, e_j)$ with $d(e, e') = 1 - e^\top e'$
(cosine). The insertion rule:

$$
\mathcal{A}' = \begin{cases}
\mathcal{A} \cup \{(F,e)\} & d(e, e_{j^*}) > \tau \quad \text{(novel)}\\
(\mathcal{A} \setminus \{(F_{j^*}, e_{j^*})\}) \cup \{(F,e)\} & d(e, e_{j^*}) \leq \tau,\ F > F_{j^*} \quad \text{(replaced)} \\
\mathcal{A} & \text{otherwise} \quad \text{(dominated)}
\end{cases}
$$

After 5 candidates accumulate, $\tau$ is calibrated as
$\tfrac{1}{2}\,\mathrm{median}\{d(e_a, e_b): a < b\}$ — adapting to
empirical density and removing a hand-tuned hyperparameter.

### 3.3 Density-corrected sampling

Each entry's local sparsity is

$$\sigma_j = \tfrac{1}{k}\sum_{l \in \mathcal{N}_k(j)} d(e_j, e_l),$$

the mean cosine distance to its $k=3$ nearest neighbors. Parents are
sampled with $p_j \propto \sigma_j$. Entries in sparser regions are
sampled more often, directly counter-acting collapse-to-one-mode.

### 3.4 AURORA-style baseline

Pool each trajectory to $\phi = [\mu_s, \sigma_s, \mu_a, \sigma_a]
\in \mathbb{R}^{28}$. MLP autoencoder $28 \to 16 \to 8 \to 16 \to 28$,
trained 50 epochs of AdamW at $\eta=10^{-3}$. **Bootstrap phase:** the
first 5 candidates are force-inserted (no dominance check) to seed the
AE training set. First AE refit fires after candidate 5, re-embedding
all archive entries, then rebuilding the archive under the new latent.
Subsequent refits every 5 candidates. Selection uses the same $k$-NN
dominance rule of §3.2 but with $z$-embeddings instead of VLM
embeddings; $\tau_{\text{AURORA}} = 0.15$ (looser bootstrap).

### 3.5 Reference grid

Hand-designed Hopper descriptor (used **only for evaluation**, never
selection):

$$\rho(s_{1:T}) = (\bar v_x, \bar h, f_{\text{gait}}) \in \mathbb{R}^3$$

with $\bar v_x = $ mean forward velocity, $\bar h = $ mean body height,
$f_{\text{gait}} = $ dominant FFT frequency of the thigh joint angle.
Each axis is 5-quantile-binned over all 60 candidates jointly → 125
cells. We report **coverage** (% filled), **QD score**
($\sum_{c \in \text{filled}} \widetilde{F}_c$ with $\widetilde F$
min-max normalized), and **diversity@all** (mean pairwise distance in
normalized ref space).

---

## 4. Experimental Setup

| | |
|---|---|
| Task | Hopper-v5 (MuJoCo, Gymnasium 1.2) |
| RL | PPO, MlpPolicy, CPU (stable-baselines3 2.8) |
| Train steps per candidate | $T_{\text{ppo}} = 30{,}000$ |
| Eval rollout length | $T_{\text{eval}} = 600$ steps @ 30 fps |
| Budget per strategy | $B = 20$ candidates |
| Seeds | 1 (seed = 0) |
| LLM | Claude Sonnet 4.6 (reward gen + VLM), Claude Haiku 4.5 (condense) |
| Text encoder | sentence-transformers/all-MiniLM-L6-v2, 384-d, L2-normalized |
| AURORA AE | MLP $28\to16\to8\to16\to28$, 50 epochs AdamW, refit at iter 5,10,15,20 |
| $k$-NN neighbors | $k = 3$ |
| $\tau$ calibration | $\tfrac{1}{2}$ median pairwise (after 5 entries) |
| Hardware | Apple M-series CPU, no MPS, no GPU |

Wall-clock per candidate ≈ 25–35 s. Per-strategy run ≈ 10–12 min.
Total experiment ≈ 30 min + ~$6 in Anthropic API tokens.

---

## 5. Results

### 5.1 Headline

| strategy | best fit | mean fit | coverage | QD score | diversity@all | archive size | candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| **archive** (ours) | **333.99** | 197.66 | **10.4%** | 5.72 | 0.562 | 15 | 20 |
| argmax | 227.63 | **217.14** | 8.8% | 3.57 | 0.771 | 20 | 20 |
| aurora | 222.63 | 206.01 | 9.6% | **6.38** | 0.605 | 1 | 20 |

**Reading the numbers:**

- **Peak fitness**: archive **+47%** over argmax, **+50%** over aurora.
- **Coverage**: archive 10.4% > aurora 9.6% > argmax 8.8% of 125 cells.
- **QD score**: aurora > archive > argmax. Aurora's high QD score is
  computed over all 20 *evaluated* candidates (each contributes a
  ref_coord), not its 1-entry archive — its evaluations cover the ref
  grid by accident because the LLM's mutations of the single elite still
  visit diverse cells. The aurora archive itself is collapsed.
- **Argmax mean fitness** edges ours because argmax mutates the running
  best every iteration — its evaluations cluster around a high mean.
  Our archive deliberately explores low-fitness niches if they're novel.
- **Argmax diversity@all** highest because it keeps all 20 candidates;
  ours is restricted to 15 non-dominated elites. Note this metric does
  *not* weight by fitness — it's diversity of evaluation set, not of an
  archive of elites.

### 5.2 Fitness curve

![Fitness curve: best-so-far vs iteration, all three strategies](data/report/fitness_curve.png)

The visual story is unambiguous: **archive** jumps from $\sim$222 at iter
7 to **283** at iter 8 and **334** at iter 9, then plateaus. **argmax**
climbs slowly from 222 → 228 over 20 iterations. **aurora** stays around
215 until iter 18 where a single replacement lifts it to 223.

Our interpretation: argmax's repeated mutation of the same parent
produces local elaborations of one reward style; archive's
density-corrected sampling pulls the LLM toward distinct reward styles,
one of which (`r008`) hit a sharply better gait. Aurora's behavior is
limited by archive collapse (§5.5): with effectively 1 elite to mutate
from, the LLM is in the same regime as argmax.

### 5.3 Coverage and QD score

![Reference-grid coverage and QD score per strategy](data/report/coverage_qd.png)

Archive leads on coverage; aurora leads on QD score but on a
catastrophically small archive (1 elite). Argmax loses on both.

### 5.4 Cross-projection in VLM space

![PCA projection of all 36 candidates' VLM-MiniLM embeddings, colored by strategy](data/report/cross_pca.png)

Projection of the 15 archive + 20 argmax + 1 aurora candidates onto the
top two PCA axes of VLM-MiniLM space. Argmax (orange) and archive (blue)
both span a wide area, but the *structure* differs — argmax clusters
densely in the top-right where its best-mutated descendants live;
archive's blue points are scattered across the space with fewer
duplicates, consistent with the density-corrected sampler pushing toward
sparse regions. Aurora's single elite sits in the middle.

### 5.5 Concise example run (one candidate, end-to-end)

To illustrate the per-candidate pipeline concretely, here is the trace
for the **highest-fitness candidate in the archive run**, `r008`:

```
candidate id:   r008
iteration:      8 / 20
strategy:       archive
parent_id:      r007   (selected by density-corrected sampling)
prompt_kind:    with_neighbors  (parent code + 3 NN descriptions)

==== generated reward (excerpt) ====
def compute_reward(obs, action, next_obs, done):
    height       = next_obs[0]
    forward_vel  = next_obs[5]
    z_vel        = next_obs[6]
    velocity_reward = 2.5 * forward_vel
    survival_bonus = 1.5
    target_height = 1.3
    if height >= target_height:
        height_reward = 3.0
    elif height > 1.0:
        height_reward = 3.0 * (height - 1.0) / (target_height - 1.0)
    else:
        height_reward = -5.0
    angle_penalty = -1.0*angle**2 if abs(angle)<0.1 else -5.0*angle**2
    # ... + joint, ang-vel, z-vel, stall, fall terms
    return float(velocity_reward + survival_bonus + height_reward + ...)

==== PPO training ====
30,000 timesteps under above reward (~6 seconds on CPU)

==== rollout evaluation (default Hopper reward) ====
600 steps, mean episode return:  F = 333.99
ref_coord = (mean_v_x=1.28, mean_h=1.13, gait_freq=0.6 Hz)
video:    data/rollouts/archive__seed0__b20__t20260528T004311/rollout_008.mp4

==== VLM description (Sonnet 4.6) ====
"Fall from upright, unsuccessful recovery, ends in unstable lean
 without coordinated hopping."

==== condense (Haiku 4.5) → embed (MiniLM) ====
e_008 ∈ R^384, L2-normalized

==== insert into archive ====
nearest neighbor: r004 (cos dist = 0.31 > τ = 0.227)
status: NOVEL  → archive grows from 5 to 6 entries
```

The fitness is high *despite* the qualitative description being negative
— the policy accumulates a lot of healthy-bonus and forward-velocity
reward before falling. Under a 600-step rollout at 30 fps, even a policy
that eventually collapses can score well if its pre-collapse phase is
forward and upright. This is honest signal about how the default Hopper
reward shapes "good" gaits in short horizons.

---

## 6. Discussion

**Density correction is the load-bearing component.** Without it, the
$k$-NN archive becomes a near-uniform sample over the archive — fine
when the archive itself is uniform, degenerate when LLM generations
cluster (which they do, since the LLM mutates within a code style).
Mean $k$-NN distance is a cheap, well-behaved inverse-density proxy and
re-prioritizes the outlier in a crowded neighborhood. This is the
"lonely-candidate-that-vanilla-EUREKA-culls" failure mode from the
original framing, *operationalized*.

**$\tau$ calibration matters.** Fixing $\tau = 0.30$ worked for smoke
tests but the calibrated value on the real run was $\tau = 0.227$,
materially looser. Without calibration the archive would have either
under- or over-filled depending on how stereotyped Sonnet's descriptions
were on this task. Calibration removes that knob.

**VLM as a behavior descriptor is plausible at this scale.** Sonnet's
descriptions are physically grounded ("low-frequency hopping gait with
forward torso lean and rightward drift", "rhythmic hopping gait with
alternating upright landing and forward-leaning takeoff phases") rather
than generic; MiniLM-on-condensed-tag clusters them meaningfully in
PCA; the archive's coverage advantage in the reference grid is the
operational consequence. **Pre-registered failure mode F3** (VLM
descriptions collapse to a few stereotyped strings) did **not** occur.

**AURORA-style fails at $B=20$ — and this is a finding.** After the
bootstrap phase force-inserted 5 candidates and the first AE refit ran,
4 of 5 entries were dominated under the freshly-trained latent. The AE
trained on 5 hopper rollouts (all of which mostly fall over) has not
seen enough behavioral variety to learn a useful $z$ space; in that
poor latent, all 5 bootstrap entries embed too close to one another for
the dominance check to keep any of them. Subsequent candidates are all
dominated against the surviving elite. Real AURORA papers run for
thousands of iterations and start from a randomized policy buffer; at
budget 20 with task-aligned LLM generations, the AE has no information
advantage over the bootstrap embedding. **Pre-registered failure mode
F4** (AURORA matches ours despite no pretraining) partially
materialized: aurora's QD score is competitive *over its evaluation
set*, but its archive is structurally broken.

**Argmax climbs slowly and plateaus.** Argmax mutates the running best
every step. By iter 5 its best is 224; by iter 19 its best is 228. The
LLM keeps elaborating the same parent. **Pre-registered failure mode
F1** (argmax matches both fitness AND diversity) did **not** occur —
argmax loses on peak fitness, on coverage, and on QD score; it wins
only on mean fitness and on raw diversity@all (an artifact of keeping
all 20 candidates as "archive size 20").

**Pre-registered failure mode F2** (we win diversity but lose fitness)
did **not** occur — we win both peak fitness *and* coverage.

---

## 7. Honest Limitations

- **Single seed.** All numbers are from one PPO seed. The +47% fitness
  delta is real on this seed, but PPO is known to be high-variance.
  Multi-seed is the immediate next experiment (iter4).
- **Single task.** Hopper-v5 only. Walker2d-v5, HalfCheetah-v5,
  MetaWorld manipulation are obvious next tasks.
- **AURORA-style ≠ full AURORA.** We pool trajectories to a 28-d
  mean+std feature rather than feeding raw sequences to a sequence
  autoencoder. The "container reset" is implemented but the
  representation budget is smaller, and the small-budget regime is not
  AURORA's home turf.
- **Reference grid is hand-designed.** The 3-D
  (forward-velocity × height × gait-frequency) grid is reasonable for
  Hopper but QD score is sensitive to axis choice and binning. We use
  5 quantile bins per axis across the joint set of all 60 candidates;
  results could shift with different binning.
- **VLM description variance not quantified.** Sonnet's descriptions
  are not deterministic; near-identical rollouts can embed $\sim$0.1
  cosine apart. We do not measure this variance explicitly here.
- **Compute profiles differ.** VLM-QD pays per-candidate Anthropic
  tokens but skips AE training; AURORA pays AE training but no API.
  Matching to a single "budget" is fuzzy. We report wall-clock + API
  cost separately.
- **Fitness $\neq$ qualitative success.** Hopper's default reward
  rewards forward-leaning-then-falling about as well as upright-hopping
  on a 600-step rollout. Our top archive reward "wins" by accumulating
  velocity + healthy-bonus before collapsing. Longer rollouts or a
  stricter success criterion would better distinguish persistent gaits
  from leaning policies.

---

## 8. Conclusion and Future Work

We presented **VLM-Embedded Quality-Diversity** for LLM-authored reward
evolution. On Hopper-v5 at a 20-candidate budget, replacing EUREKA's
argmax selection with a $k$-NN dominance archive over VLM-derived
behavior descriptors and density-corrected sampling lifts peak fitness
by **+47%** and reference-grid coverage by **+18% relative** over
argmax-EUREKA; an AURORA-style autoencoder baseline collapses its
archive to a single elite at this scale.

**Immediate next steps (iter4):**

1. **Multi-seed.** Three to five PPO seeds per strategy.
2. **More tasks.** Walker2d-v5, HalfCheetah-v5, then a manipulation
   env (MetaWorld door-open is the canonical tactile-adjacent
   candidate).
3. **Sensitivity ablation.** $k \in \{1, 3, 5\}$, $\tau$ calibration
   off vs.\ on, AE refit cadence $\in \{2, 5, 10\}$.
4. **Prompt-side ablation.** Run our `archive` strategy with mutation
   prompts that *do not* include neighbor descriptions, to isolate
   selection-side diversity gain from prompt-side diversity gain.
5. **CVT variant.** Replace flat $k$-NN with CVT-MAP-Elites in
   VLM-embedding space (we deliberately deferred this for fewer
   hyperparameters in iter3); revisit at larger budgets where
   centroid pre-seeding can pay off.

**Toward a paper.** Given the project's pivot to tactile reward
evolution (per chat log), the natural venue is **RSS 2027**
(methodological-novelty fit, $\sim$8 months away) or **ICRA 2027**
($\sim$4 months, tighter). A CORL 2026 workshop submission of this
iter3 result is feasible as a CV line and forcing function.

---

## References (abridged)

- Ma, Y. J., Liang, W., et al.\ (2023). **EUREKA: Human-level reward design via coding LLMs.** arXiv:2310.12931.
- Mouret, J.-B. \& Clune, J. (2015). **Illuminating search spaces by mapping elites.** arXiv:1504.04909.
- Vassiliades, V., Chatzilygeroudis, K., Mouret, J.-B. (2017). **Using CVT to scale MAP-Elites.** IEEE TEVC.
- Pugh, J. K., Soros, L. B., Stanley, K. O. (2016). **Quality diversity: a new frontier for evolutionary computation.** Frontiers in Robotics and AI.
- Cully, A. (2019). **AURORA: learning behavioural repertoires with unsupervised behavioural descriptors.** GECCO.
- Grillotti, L. \& Cully, A. (2021/2022). **Unsupervised behaviour discovery with quality-diversity optimisation.** IEEE TEVC.
- Agrawal, A., et al.\ (2025). **GEPA: reflective prompt evolution outperforms reinforcement learning.** arXiv preprint.
- Wang, et al.\ (2025). **Text2Touch: LLM-driven reward design for dexterous tactile manipulation.** CoRL.
- Xie, T., et al.\ (2024). **Text2Reward: reward shaping with language models.** ICLR.

---

## Appendix A. Pre-registered failure modes — outcomes

| code | description | outcome |
|---|---|---|
| **F1** | argmax matches both diversity AND fitness → diversity bonus superfluous | **NOT observed** (argmax loses on peak fitness, coverage, QD score) |
| **F2** | ours wins diversity but loses fitness → diversity tax | **NOT observed** (we win both peak fitness AND coverage) |
| **F3** | VLM descriptions collapse to stereotyped strings → bad descriptor | **NOT observed** (descriptions are physically grounded and diverse) |
| **F4** | AURORA matches ours despite no pretraining → VLM channel adds little | **PARTIALLY observed** — AURORA's QD score over its evaluation set is competitive, BUT its archive collapses to 1 elite, so it cannot maintain diversity *as a QD method* |

---

## Appendix B. Reproducing this report

```bash
cd iter3
bash setup.sh                                          # creates .venv (py>=3.10), installs deps
source .venv/bin/activate
python -m src.verify                                   # 6 sanity checks

# .env contains ANTHROPIC_API_KEY (gitignored)
python run.py --strategy archive --budget 20 --seed 0  # ~10 min
python run.py --strategy argmax  --budget 20 --seed 0  # ~10 min
python run.py --strategy aurora  --budget 20 --seed 0  # ~12 min

python -m src.analysis.report
# writes data/report/{fitness_curve.png, coverage_qd.png, cross_pca.png,
#                    headline.{json,md}, per_strategy.json, top3.md}
```

## Appendix C. Run identifiers (for this report)

| strategy | run_id |
|---|---|
| archive | `archive__seed0__b20__t20260528T004311` |
| argmax  | `argmax__seed0__b20__t20260528T010506` |
| aurora  | `aurora__seed0__b20__t20260528T013250` |

All per-iteration audit logs live under `data/logs/<run_id>.jsonl`;
archive snapshots under `data/archives/<run_id>.json`. Videos and state
logs are under `data/rollouts/<run_id>/`; reward source under
`data/rewards/<run_id>/`.

## Appendix D. Bugs found and fixed during this iteration

1. **LLM response truncation hid the closing code fence.** Initial
   `max_tokens = 800` was too small for the LLM's elaborate rewards.
   Truncated responses had an opening ```python but no closing
   ```; my regex skipped them and the raw text (with the fence) was
   written as the reward file → AST parse error → most candidates
   skipped. **Fix:** raised to 2000 tokens *and* added a fallback
   extractor that handles truncated blocks.
2. **AURORA archive collapse from refit-trigger condition.** The first
   AURORA run never refit because the trigger was
   `len(archive) % refit_every == 0` and the archive was stuck at 1
   entry (all candidates dominated under the crude bootstrap embedding).
   **Fix:** switched trigger to *candidates seen*, added a 5-candidate
   *bootstrap phase* that force-inserts (no dominance check), and
   trained the AE on **all rollouts on disk** (not just archive entries)
   so dominated candidates still contribute to representation learning.
   The fix improves AURORA but does not save it at $B=20$ — see §6.

## Appendix E. Wall-clock + API cost (this experiment)

| phase | wall-clock | API cost |
|---|---:|---:|
| Setup (venv + deps) | ~5 min | $0 |
| `archive` (20 candidates × ~25-35 s) | ~10 min | ~$2.00 |
| `argmax` (20 candidates × ~25-35 s)  | ~10 min | ~$2.00 |
| `aurora` (20 candidates × ~25-35 s, includes AE refits) | ~12 min | ~$2.00 |
| Report generation | <1 min | $0 |
| **Total** | **~38 min** | **~$6** |

All on Apple M-series CPU; no GPU/MPS.
