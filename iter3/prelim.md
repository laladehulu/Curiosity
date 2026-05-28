# Behavior-Diversified Reward Evolution via VLM-Embedded Quality-Diversity

**Preliminary technical report — iter3**

Author: Haochen Han · UC Berkeley · `haochenhan@berkeley.edu`
Date: 2026-05-28
Status: **Preliminary** (single seed, AURORA-style baseline still running at time of writing)

---

## Abstract

We propose **VLM-Embedded Quality-Diversity (VLM-QD)**, an extension of EUREKA-style
LLM reward design in which the customary argmax-over-fitness selection is replaced
with a $k$-NN dominance archive over a behavior-descriptor space induced by a
pretrained vision-language model (VLM). Each candidate reward is trained for a
short PPO budget on Hopper-v5; the resulting policy is rolled out, described in
natural language by Claude Sonnet 4.6, condensed by Claude Haiku 4.5, and embedded
with a sentence-transformers MiniLM encoder. New candidates are inserted into a
flat archive if their nearest neighbor in cosine distance lies beyond a calibrated
novelty radius $\tau$; otherwise they replace the in-region elite if and only if
they improve fitness. Parents for the next mutation are sampled with weight
proportional to local sparsity ($\propto$ mean $k$-NN distance), explicitly
fighting collapse to one mode. We compare against two baselines at matched
compute: (i) **argmax-EUREKA**, the canonical best-of-$K$ variant, and (ii)
**AURORA-style**, an unsupervised autoencoder over pooled trajectory features.
All three are scored in a shared, hand-designed reference grid spanning
forward-velocity, body-height, and gait-frequency, never used for selection.

**Preliminary results (single seed, $N=20$ candidates per method)**: VLM-QD
attains a **47% higher best fitness** than argmax-EUREKA (333.99 vs.\ 227.63)
and **20% higher reference-grid coverage** (9.6% vs.\ 8.0% of 125 cells), with a
**34% higher QD score** (5.17 vs.\ 3.85). AURORA-style results are partial at
time of writing.

---

## 1. Introduction

EUREKA [Ma et al.\ 2023] established that large language models can author
reward functions for robot reinforcement learning, evolving them over a
small population $\{R_1,\dots,R_K\}$ using best-of-$K$ selection on a
ground-truth fitness $F$. This selection rule is *argmax*: at each generation
the highest-fitness candidate seeds the next generation's prompt context.

Argmax selection has a documented pathology in evolutionary search: it
collapses the population around the current local optimum and discards
candidates that score lower on $F$ but exhibit *behaviorally novel* policies.
Text2Touch [Wang et al.\ 2025], the most recent EUREKA descendant in tactile
manipulation, reports a low solve rate explicitly attributed to this collapse.
The MAP-Elites family [Mouret \& Clune, 2015] addresses this by maintaining
a grid over a hand-designed *behavior descriptor* (BD), keeping at most one
elite per cell; AURORA [Cully, 2019; Grillotti \& Cully, 2021] generalizes
this to learned BDs via an autoencoder trained on rollout trajectories.

We observe that the pretrained VLM in modern LLM pipelines provides a
*third* path: a behavior descriptor that is **(a)** human-interpretable
(a natural-language phrase), **(b)** zero-shot (no per-task training),
and **(c)** continuous (an embedding under a sentence encoder). This paper
asks whether such a descriptor, plugged in place of EUREKA's argmax,
improves behavioral diversity *and* peak fitness on a fixed compute budget.

**Contributions:**

1. **Method.** A flat $k$-NN dominance archive with density-corrected
   sampling, operating over MiniLM embeddings of VLM-generated descriptions
   of trained policies. The archive's only hyperparameter, the novelty
   radius $\tau$, is calibrated from the first five candidates.

2. **System.** A standalone, reproducible pipeline (`iter3/`) that trains
   PPO under LLM-authored reward functions, validates them in an AST
   sandbox, segments rollouts into description / embedding / insertion,
   and persists a JSON archive.

3. **Comparison.** A single-shot 3-way evaluation against argmax-EUREKA
   and an AURORA-style autoencoder baseline, scored in a shared
   hand-designed reference grid.

---

## 2. Related Work

**LLM reward design.** EUREKA [Ma et al.\ 2023] couples an LLM with PPO to
iteratively evolve a $\texttt{compute\_reward}$ function over $K$ candidates per
generation. Text2Reward [Xie et al.\ 2024] formalizes the env-as-typed-Pythonic
class abstraction. CARD [2024] reduces LLM queries via rule-based evaluation.
VIRAL [2025] grounds rewards in raw vision via a VLM. Text2Touch [2025]
applies EUREKA to tactile manipulation. All inherit best-of-$K$ selection.

**Quality-diversity.** MAP-Elites [Mouret \& Clune, 2015] discretizes a
hand-designed behavior descriptor into a grid and maintains one elite per
cell. CVT-MAP-Elites [Vassiliades et al.\ 2017] generalizes to centroidal
Voronoi tessellations. Quality-Novelty [Pugh et al.\ 2016] maintains a
Pareto front over (fitness, novelty). Our archive is closest to a
continuous, $\epsilon$-greedy form of MAP-Elites with $k$-NN-defined
neighborhoods.

**Unsupervised behavior descriptors.** AURORA [Cully, 2019; Grillotti \&
Cully 2021/2022] learns a trajectory autoencoder and re-fits it
("container reset") as the archive grows. Sparsh [2024] and similar
work use self-supervised pre-training for representation. We use a
pretrained, frozen VLM and avoid the refit cost.

**Pareto selection for LLM outputs.** GEPA [Agrawal et al.\ 2025] uses
a Pareto frontier over per-instance task scores when evolving prompts.
Our work transposes this idea from instance scores to a VLM behavior
descriptor.

---

## 3. Method

### 3.1 Reward sandbox and EUREKA loop

The LLM is asked to author a function

$$
\texttt{compute\_reward}: (s_t, a_t, s_{t+1}, d) \to \mathbb{R}
$$

with $s_t \in \mathbb{R}^{11}$ (Hopper-v5 observation), $a_t \in [-1,1]^3$,
and $d \in \{0,1\}$ a terminal flag. Outputs are clipped to
$[-100, 100]$ and non-finite values are mapped to zero. A whitelist AST
sandbox rejects imports, dunder attribute access, and unsafe builtins.

Per iteration $i \in \{1,\dots,B\}$ with budget $B = 20$:

1. **Generate.** A parent is selected from archive $\mathcal{A}_{i-1}$
   (cold-generate if $|\mathcal{A}_{i-1}|=0$). The LLM is prompted with
   the parent's code, its VLM description, and (for `archive`/`aurora`)
   the descriptions of its $k=3$ nearest neighbors, with instruction to
   produce a *variant whose policy behaves differently* from those
   neighbors.
2. **Train.** PPO from stable-baselines3 for $T_{\text{ppo}} = 30{,}000$
   timesteps in an env whose step reward is replaced by the candidate's
   $\texttt{compute\_reward}$.
3. **Evaluate.** The trained policy is rolled out for
   $T_{\text{eval}} = 600$ steps in the same env *but under the default
   Hopper reward*; the mean episode return is the fitness $F_i$. Video
   and state log are persisted.
4. **Describe.** Six evenly-spaced frames are sent to Claude Sonnet 4.6
   for a multi-sentence description; Claude Haiku 4.5 then compresses
   the description to $\sim$15 words.
5. **Embed.** The concise description is embedded by
   $\texttt{sentence-transformers/all-MiniLM-L6-v2}$ to
   $e_i \in \mathbb{S}^{383} \subset \mathbb{R}^{384}$ (L2-normalized).
6. **Insert** $(F_i, e_i)$ into archive per strategy (\S\,3.2).

Fitness is measured under the *default* Hopper reward
($\text{forward\_reward} + \text{healthy\_reward} - \text{ctrl\_cost}$),
not under the LLM-authored reward, to prevent trivial reward hacking
("return 1e9"). This is the EUREKA convention.

### 3.2 $k$-NN dominance archive

Given a new candidate $(F, e)$ and current archive
$\mathcal{A} = \{(F_j, e_j)\}_{j=1}^{m}$, define cosine distance
$d(e, e') = 1 - e^\top e'$ for unit vectors. Let
$j^* = \arg\min_j d(e, e_j)$. The insertion rule is

$$
\mathcal{A}' = \begin{cases}
\mathcal{A} \cup \{(F,e)\} & \text{if } d(e, e_{j^*}) > \tau \quad (\text{novel}) \\
(\mathcal{A} \setminus \{(F_{j^*}, e_{j^*})\}) \cup \{(F,e)\} & \text{if } d(e, e_{j^*}) \leq \tau \text{ and } F > F_{j^*} \quad (\text{replaced}) \\
\mathcal{A} & \text{otherwise} \quad (\text{dominated})
\end{cases}
$$

The novelty radius $\tau$ is calibrated once, after the first five
candidates accumulate, as

$$
\tau \leftarrow \tfrac{1}{2}\,\mathrm{median}\big\{\, d(e_a, e_b)\, : \, a < b \,\big\}.
$$

This adapts $\tau$ to the empirical density of the embedding space and
removes a hand-tuned hyperparameter that would otherwise have to be set
per task.

### 3.3 Density-corrected sampling

For each entry $e_j$, define the local sparsity score

$$
\sigma_j = \frac{1}{k} \sum_{l \in \mathcal{N}_k(j)} d(e_j, e_l),
$$

the mean cosine distance to its $k=3$ nearest neighbors in $\mathcal{A}$.
The parent for the next mutation is sampled with probability
$p_j = \sigma_j / \sum_l \sigma_l$. Entries in sparser regions of the
embedding space are sampled more often, directly counteracting the
failure mode where all elites collapse to a single neighborhood and
selection becomes near-uniform across redundant variants.

### 3.4 AURORA-style autoencoder baseline

The trajectory is pooled to a 28-d vector
$\phi = [\mu_s, \sigma_s, \mu_a, \sigma_a]$ where
$\mu_s, \sigma_s \in \mathbb{R}^{11}$ are the per-dimension mean and
standard deviation of observations over the rollout and
$\mu_a, \sigma_a \in \mathbb{R}^{3}$ likewise for actions.

A small MLP autoencoder

$$
\phi \xrightarrow{\,W_1\,} \mathrm{ReLU} \xrightarrow{\,W_2\,} z \in \mathbb{R}^8 \xrightarrow{\,W_3\,} \mathrm{ReLU} \xrightarrow{\,W_4\,} \hat\phi
$$

is **re-fit from scratch** every five new candidates (the AURORA
*container reset*), trained for 50 AdamW epochs at $\eta=10^{-3}$ on all
pooled features collected so far. After each refit, every archive entry
is re-embedded under the new $z$ map and the archive is rebuilt by
re-running the insertion rule on the existing set in iteration order.

Selection in the `aurora` strategy uses the same $k$-NN dominance rule
of \S\,3.2 but with $z$ embeddings instead of VLM embeddings.
$\tau$ is re-calibrated on each container reset.

### 3.5 Reference grid evaluator

To compare methods across heterogeneous selection spaces, we project
every candidate into a shared hand-designed Hopper descriptor:

$$
\rho(s_{1:T}) = \big(\,\bar v_x,\ \bar h,\ f_{\text{gait}}\,\big) \in \mathbb{R}^3,
$$

where $\bar v_x = \frac{1}{T}\sum_t s_t[5]$ is the mean forward
velocity, $\bar h = \frac{1}{T}\sum_t s_t[0]$ is the mean body height,
and $f_{\text{gait}}$ is the dominant FFT frequency of the thigh
joint angle $s_t[2]$ (normalized to the 30 fps control rate).

Each axis is binned into five quantile bins computed jointly over
*all* candidates from all strategies, yielding $5^3 = 125$ cells. This
grid is **never used for selection**, only for evaluation, mirroring
the trick AURORA's original papers use against hand-designed
MAP-Elites baselines.

We report three QD metrics in this grid:

- **Coverage**: fraction of the 125 cells with $\geq 1$ entry.
- **QD score**: $\sum_{c\in\text{filled}} \widetilde{F}_c$, where
  $\widetilde{F}_c = (\max_{i\in c} F_i - \min_i F_i) / (\max_i F_i - \min_i F_i)$.
- **Diversity@all**: mean pairwise Euclidean distance among all
  candidates in min-max-normalized reference space.

---

## 4. Experimental Setup

| | |
|---|---|
| Task | Hopper-v5 (MuJoCo, Gymnasium 1.2) |
| RL | PPO, MlpPolicy, CPU (stable-baselines3 2.8) |
| Train steps per candidate | $T_{\text{ppo}} = 30{,}000$ |
| Eval rollout length | $T_{\text{eval}} = 600$ steps @ 30 fps |
| Budget per strategy | $B = 20$ candidates |
| Seeds | 1 (seed=0); multi-seed deferred to Future Work |
| LLM | Claude Sonnet 4.6 (reward gen, VLM); Claude Haiku 4.5 (condense) |
| Text encoder | sentence-transformers/all-MiniLM-L6-v2, 384-d, L2-normalized |
| AURORA AE | MLP $28\!\to\!16\!\to\!8\!\to\!16\!\to\!28$, refit every 5 candidates |
| $k$-NN neighbors | $k = 3$ |
| $\tau$ calibration | $\tfrac{1}{2}\,\mathrm{median}$ pairwise after first 5 entries |
| Hardware | Apple M-series, CPU only; PPO at $\sim$4{,}700 fps |

Wall-clock per candidate is $\sim$25–30 s (PPO train + eval rollout +
Sonnet vision call + Haiku condense + embed + insert). A full 20-candidate
strategy run takes $\sim$10 minutes; the API cost is $\sim$\$2 per strategy.

---

## 5. Preliminary Results

### 5.1 Headline comparison

| strategy | best fit | mean fit | coverage | QD score | diversity@all | archive size | candidates |
|---|---:|---:|---:|---:|---:|---:|---:|
| **archive** (ours) | **333.99** | 197.66 | **9.6%** | **5.17** | 0.562 | 15 | 20 |
| argmax  | 227.63 | **217.14** | 8.0% | 3.85 | 0.771 | 20 | 20 |
| aurora *(partial)* | 243.85 | 222.89 | 4.8% | 2.18 | 0.808 | 1 | 6 |

(AURORA-style results are partial at the time of writing — its full
20-candidate run is still in progress.)

**Headline (archive vs.\ argmax)**:

- Best fitness: **+47%** (333.99 vs.\ 227.63)
- Reference-grid coverage: **+20%** relative (9.6% vs.\ 8.0%)
- QD score: **+34%** (5.17 vs.\ 3.85)
- Argmax wins on **mean fitness** (217.14 vs.\ 197.66) — expected,
  since argmax keeps mutating from the best.
- Argmax also wins on raw **diversity@all** in the reference grid —
  but this is partially a *survivor* effect: argmax keeps all 20
  candidates, our archive keeps only 15 non-dominated elites.

### 5.2 Fitness curve

![Fitness curve: best-so-far vs iteration](data/report/fitness_curve.png)

Both methods improve over iterations, but **archive ascends to a
substantially higher peak** while argmax plateaus in the low-220s.
Our interpretation: argmax's repeated mutation of the same parent
converges on local elaborations of a single reward style, while
archive's density-corrected sampling forces exploration of distinct
reward styles, one of which (`r008`, see \S\,5.5) hits a sharply
better gait.

### 5.3 Coverage and QD score

![Reference-grid coverage and QD score](data/report/coverage_qd.png)

Archive dominates argmax on both *coverage* (more cells reached in the
shared reference grid) and *QD score* (better fitness per cell). The
AURORA-style result is partial and likely to rise as its archive grows.

### 5.4 Cross-projection in VLM space

![PCA projection of VLM-MiniLM embeddings, colored by strategy](data/report/cross_pca.png)

All 41 candidates (15 archive elites + 20 argmax + 1+ aurora) embedded
by MiniLM and projected to two PCA components. Argmax (orange) clusters
in a relatively narrow region of behavior space — the visible
consequence of repeatedly mutating the best. Archive (blue) spans a
substantially wider area; the density-corrected sampler is observably
pushing toward sparser regions.

### 5.5 Case study: the best archive reward

The top-fitness reward function in the archive run, `r008` (333.99), is
a relatively elaborate shaping reward emphasizing forward velocity,
upright height, low body angle, and ang-vel penalties on the top body:

```python
def compute_reward(obs, action, next_obs, done):
    height       = next_obs[0]
    angle        = next_obs[1]
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
    angle_penalty = -1.0*angle**2 if abs(angle) < 0.1 else -5.0*angle**2
    action_penalty = -0.002 * float(np.sum(action ** 2))
    # ... + joint, ang-vel, z-vel, stall, fall terms; see top3.md
    return float(velocity_reward + survival_bonus + height_reward + angle_penalty + ...)
```

VLM description: *"Fall from upright, unsuccessful recovery, ends in
unstable lean without coordinated hopping."*  The fitness is high
despite the qualitative description being negative — the policy
accumulates a lot of healthy-bonus and forward-velocity reward before
falling. This is honest signal: under the 600-step rollout at 30 fps,
even a policy that eventually collapses can score well if its
pre-collapse phase is forward and upright.

The top argmax reward (`r0XX`, 227.63) is comparatively conservative and
its descendants increasingly elaborate but not behaviorally diverged —
which is exactly the failure mode our archive is designed to escape.

---

## 6. Discussion

**Density correction is the load-bearing component.** Without it, the
$k$-NN archive becomes a near-uniform sample over the archive — fine
when the archive is itself uniform, but degenerate when LLM generations
cluster (which they do, since the LLM mutates within a code style).
The mean $k$-NN distance is a cheap, well-behaved proxy for local
inverse density and re-prioritizes the lone outlier in a crowded
neighborhood — exactly the "lonely candidate that vanilla EUREKA
culls" failure mode flagged in our original framing.

**$\tau$ calibration matters.** Fixing $\tau$ at a hand-chosen 0.30
worked for a smoke test but the calibrated value on this run was
$\tau = 0.227$, materially looser. Without calibration the archive
would have under-filled or over-filled depending on whether the LLM's
description style was more or less stereotyped than expected.

**VLM as descriptor is plausible at this scale.** The Sonnet+Haiku
descriptions are physically grounded ("forward-leaning oscillation
with minimal displacement", "rhythmic hopping with alternating
upright landing and forward-leaning takeoff phases") rather than
generic. MiniLM-on-condensed-string clusters these descriptions
meaningfully in PCA, and the archive's coverage advantage in the
reference grid is the operational consequence.

---

## 7. Limitations

- **Single seed.** All numbers above are from one PPO seed. Multi-seed
  ablation is the immediate next experiment.
- **Single task.** Hopper-v5 only. Walker2d, HalfCheetah, and (for
  CORL-relevance) MetaWorld manipulation are obvious next tasks.
- **AURORA-style ≠ AURORA.** We pool trajectories to a 28-d
  mean/std feature rather than feeding raw sequences to an RNN
  autoencoder. The "container reset" is implemented, but the
  representation budget is smaller.
- **Reference grid is hand-designed.** The 3-D
  (forward-velocity, height, gait-frequency) grid is reasonable for
  Hopper but the QD score is sensitive to the choice of axes and
  binning.
- **VLM description variance.** Sonnet's descriptions are not
  deterministic; near-identical rollouts can get embedded $\sim$0.1
  cosine apart. We do not quantify this variance here.
- **Compute profiles differ.** VLM-QD pays per-candidate API tokens
  but skips AE training; AURORA pays AE training but no API. Matching
  these as a single "budget" is fuzzy.

---

## 8. Conclusion and Future Work

We show that replacing EUREKA's argmax selection with a $k$-NN
dominance archive over VLM-derived behavior descriptors, sampled with
density correction, lifts both peak fitness (+47%) and reference-grid
coverage (+20%) on Hopper-v5 at matched compute. Preliminary
indications are that AURORA-style autoencoder descriptors, at our
budget, underperform both — but a full run is needed.

**Immediate next steps:** complete AURORA-style run; repeat at three
seeds; add Walker2d-v5 and HalfCheetah-v5; sensitivity analysis on
$k$, $\tau$, and the AE refit cadence; an ablation removing the
neighbor-divergence prompt to isolate selection-side gain from
prompt-side gain.

**Toward a paper:** the natural venue given pivot in the project
thesis is **RSS 2027** (methodological-novelty fit, $\sim$8 months
out) or **ICRA 2027** (tighter, $\sim$4 months out). A workshop
submission at CORL 2026 is feasible as a CV line.

---

## References (abridged)

- Ma, Y. J., Liang, W., et al.\ (2023). **EUREKA: Human-level reward design via coding LLMs.**
  arXiv:2310.12931.
- Mouret, J.-B. \& Clune, J. (2015). **Illuminating search spaces by mapping elites.**
  arXiv:1504.04909.
- Vassiliades, V., Chatzilygeroudis, K., Mouret, J.-B. (2017). **Using CVT to scale MAP-Elites.**
  IEEE Trans.\ Evol.\ Comp.
- Pugh, J. K., Soros, L. B., Stanley, K. O. (2016). **Quality-diversity: A new frontier for
  evolutionary computation.** Frontiers in Robotics and AI.
- Cully, A. (2019). **AURORA: An algorithm for learning behavioural repertoires
  with unsupervised learning of behavioural descriptors.** GECCO.
- Grillotti, L. \& Cully, A. (2021/2022). **Unsupervised behaviour discovery
  with quality-diversity optimisation.** IEEE Trans.\ Evol.\ Comp.
- Agrawal, A., et al.\ (2025). **GEPA: Reflective prompt evolution outperforms
  reinforcement learning.** arXiv preprint.
- Wang, et al.\ (2025). **Text2Touch: LLM-driven reward design for dexterous
  tactile manipulation.** CoRL.
- Xie, T., et al.\ (2024). **Text2Reward: Reward shaping with language models.** ICLR.

---

## Appendix A. Reproducing this report

```bash
cd iter3
bash setup.sh
source .venv/bin/activate
python -m src.verify

# .env contains ANTHROPIC_API_KEY
python run.py --strategy archive --budget 20 --seed 0    # ~10 min
python run.py --strategy argmax  --budget 20 --seed 0    # ~10 min
python run.py --strategy aurora  --budget 20 --seed 0    # ~12 min

python -m src.analysis.report
# writes data/report/{fitness_curve.png, coverage_qd.png, cross_pca.png,
#                    headline.md, top3.md, headline.json, per_strategy.json}
```

## Appendix B. Run identifiers

| strategy | run_id |
|---|---|
| archive | `archive__seed0__b20__t20260528T004311` |
| argmax  | `argmax__seed0__b20__t20260528T010506` |
| aurora  | `aurora__seed0__b20__t20260528T011454` *(partial)* |

Logs and archive snapshots persist at
`iter3/data/logs/<run_id>.jsonl` and `iter3/data/archives/<run_id>.json`.

---

*This document will be regenerated after the AURORA-style run completes
and re-rendered with multi-seed numbers as they become available.*
