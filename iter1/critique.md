# Self-critique — iter1 (PARTIAL — scaffold-time)

Per protocol §"Phase 6: Analysis and writeup → `critique.md`".

This file is split into **(A) scaffold-time critique** (known issues with the
code that Claude Code wrote, written before any experiment runs) and
**(B) experiment-time critique** (to be filled after Phase 5).

---

## A. Scaffold-time critique

### A.1 GEPA library was not used

The protocol's §"Setup phase" lists `pip install -e .` from
https://github.com/gepa-ai/gepa as a dependency. The actual GEPA library is
designed for *prompt* optimization (its adapters expect text I/O), not reward
search. Rather than monkey-patching its adapters in fragile ways,
`reward_gen/gepa_pareto.py` reimplements the conceptual recipe —
non-dominated sort over per-instance score vectors, plus reflective LLM
mutation — directly. This is a knowing deviation. Reviewers asking "is this
really GEPA?" should be answered honestly: it is *GEPA's selection and
reflective-mutation principles applied to reward code*, not a wrapper on the
GEPA library.

If the human author needs library-level lineage to the GEPA paper, the right
fix is to subclass `gepa.GEPAAdapter` and route reward-search through it.
That subclass is unwritten.

### A.2 LLM-generated reward safety is shallow

`reward_template._validate_ast` blocks the obvious bad imports (`os`, `sys`,
`subprocess`, `socket`) and the obvious bad calls (`open`, `exec`, `eval`,
`compile`, `__import__`). It does NOT defend against:

- Resource exhaustion (infinite loops, huge tensor allocations) — protocol
  scope is workshop pilot, not adversarial
- Subtle numerical pathologies that produce NaN-then-clip cascades
- Side-effects through numpy mutation of input arrays

These are acceptable for the workshop scope but should be flagged if the work
is extended.

### A.3 BipedalWalker descriptor uses observation-space proxies

The "gait frequency" descriptor estimates oscillation from hip1 zero-crossings.
This is a known-noisy proxy and will conflate stride frequency with
balance-recovery wobbles. The protocol allows simple descriptors (workshop
scope) but the noise will eat some archive resolution.

### A.4 MountainCar descriptor deviates from protocol

Protocol says "max position reached **before first success**". For policies
that never succeed (likely most early candidates), "before first success"
is undefined. We use max position over the whole episode regardless. Document
this in the paper.

### A.5 Phase 1 short-PPO is a noisy fitness signal

Per protocol §"Score each candidate reward by training a short PPO (50K steps)
under it". 50K PPO steps with 1 seed is *very* noisy for Phase 1 selection.
Both variants suffer this equally so the comparison is fair, but absolute
quality of selected pools will be lower than 5-seed selection. This is a
direct protocol parameter — do not change without protocol update.

### A.6 Reflective mutation prompt is generic

`build_mutation_user_msg` shows the LLM the parent code, training curve, and
fitness summary. It does NOT show:
- Other candidates from the pool (only the parent)
- A diff between parent and rejected siblings
- Failure traces of compile failures

Eureka-style "evolutionary signal" is therefore weaker than the original
EUREKA paper, which exposes more population context. This is a scope-trade.

### A.7 Box2D / BipedalWalker robustness perturbations are best-effort

The Box2D world is recreated on each `env.reset()`, so attribute mutations
applied to `env.unwrapped` may be wiped. The setup.sh script and
`perturbations.py` re-apply mutations after reset, but verifying this
actually changes physics requires inspection of Box2D fixtures — not done at
scaffold time. If BipedalWalker robustness results look suspiciously
invariant to perturbation, this is the first place to look.

### A.9 LLM swapped from OpenAI to Anthropic Claude (user request)

The protocol calls for `gpt-4o-mini` / `gpt-4o`. Per user request 2026-05-21
we swapped to Anthropic's Messages API: `claude-haiku-4-5` for the testing
slot (~$1/$5 per 1M tokens, equivalent role to `gpt-4o-mini`) and
`claude-opus-4-7` available for finals via `--model` (~$5/$25 per 1M).
Functionally equivalent — both are LLMs producing reward-function code from
the same prompts — but the paper should note the deviation, since LLM choice
affects generated-reward quality and is not a controlled variable.

Prompt-caching markers are placed on the stable task prompt block per
`shared/prompt-caching.md`. Our prompts are well below the 4096-token
cacheable minimum on both haiku-4-5 and opus-4-7, so the markers will
**silently no-op** in this workload. Cost of having them is zero, and they
will start firing automatically if reflection contexts grow past the minimum.

### A.8 No GEPA-library cost-aware mutation

Real GEPA budgets LLM calls against per-candidate gain. Our budget is flat:
50 calls regardless of progress. Not a flaw vs the protocol, but worth noting
that real GEPA's anytime properties are not in scope here.

---

## B. Experiment-time critique (TBD)

To be filled after Phase 5:

### B.1 Threats to validity

_TBD._

### B.2 What we did not measure

_TBD._

### B.3 What we should have done with more compute

_TBD._

### B.4 Possible reviewer objections

_TBD — anticipated and responded to._

### B.5 Comparison to prior work that we didn't run

Out-of-scope per protocol:
- Eureka (no Isaac Gym; no fair comparison available)
- DrEureka (sim-to-real; out of scope)
- RoboMoRe (manipulation; out of scope)

Per protocol §"Comparison to prior work that we didn't run", we acknowledge
these gaps directly rather than pretending they don't matter.
