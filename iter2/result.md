# iter2 — Results

End-to-end behavior-indexing pipeline run on 2026-05-27.

## Pipeline executed

```
PPO training  ->  rollout (mp4 + npz)  ->  change-point segmentation
   ->  Claude Sonnet 4.6 (vision, detailed)  ->  Claude Haiku 4.5 (condense)
     ->  sentence-transformers all-MiniLM-L6-v2 (384-d)  ->  ChromaDB (cosine)
```

## Artifacts produced

| Stage | Count | Notes |
|---|---|---|
| Checkpoints | 9 | 3 envs × 3 training-step snapshots each |
| Rollout videos | 9 mp4 | 600 steps @ 30 fps, ~3.5 MB each |
| State logs | 9 npz | obs / action / reward / terminated / truncated |
| Segments | 41 | from `ruptures` PELT + episode-reset + max-split |
| VLM descriptions | 41 | detailed (Sonnet) + concise (Haiku) |
| Chroma entries | 41 | persistent at `data/chroma/` |

## Compute / wall-clock

| Stage | Time |
|---|---|
| Training all 9 checkpoints | **~3 min** (300k Walker2d in 66s, 200k Hopper in 42s, 300k HalfCheetah in 54s — all on M-series CPU, no MPS) |
| 9 rollouts + segmentation | ~20 s total |
| 41 VLM calls (Sonnet + Haiku) | ~7 min (avg ~8–10 s per Sonnet vision call, ~2 s per Haiku condense) |
| Embeddings + Chroma upsert | < 5 s (MiniLM is cheap, batched) |
| **Total end-to-end** | **~11 min** |

## Catalog

```
Walker2d-v5    300k steps   checkpoints @ 50k / 150k / 300k
Hopper-v5      200k steps   checkpoints @ 40k / 100k / 200k
HalfCheetah-v5 300k steps   checkpoints @ 50k / 150k / 300k
```

Three checkpoints per env to capture an emergent-behavior gradient (early =
stumbling/falling, late = closer to a stable gait) from a single training run.

## Bugs encountered and fixes

1. **`setup.sh` picked the system Python 3.9 from `/Library/Developer/CommandLineTools/`.**
   `python3.12` was unavailable so the script fell back to `python3`, which
   resolved to Apple's CLT python in the non-interactive shell. `mujoco` ships
   no 3.9 wheel, so install died on `mujoco`'s source build (`MUJOCO_PATH not
   set`).
   **Fix:** rewrote interpreter selection to scan candidates and require
   `python_version >= 3.10`. Verified end-to-end with Python 3.14.5.

2. **`gymnasium[mujoco]` env IDs were on the deprecated `-v4` tier.**
   Gymnasium warned about v4 being out-of-date; v5 is the current canonical.
   **Fix:** updated `catalog.py`, `verify.py`, and README to use v5.

3. **Chroma 1.x calls `embed_query` / `embed_documents` on the embedding
   function**, not just `__call__`. The first search raised
   `AttributeError: 'ChromaEmbeddingFunction' object has no attribute
   'embed_query'`.
   **Fix:** added `embed_query` and `embed_documents` methods to
   `ChromaEmbeddingFunction` (both route to the same `embed()` since
   sentence-transformers doesn't differentiate queries from documents).

4. **`ANTHROPIC_API_KEY` not reachable from non-interactive shell.**
   Pasting it once into a chat session doesn't propagate to spawned bash
   processes.
   **Fix:** added `src/__init__.py` side-effect that loads `iter2/.env` into
   `os.environ` at module import, and a `.gitignore` to keep the key file
   out of git.

## Sample VLM outputs (concise strings, by checkpoint)

```
walker2d 300k  seg0  Walker2d collapsing rightward with tilting torso and trailing foot.
walker2d 300k  seg3  Walker2d falls forward uncontrolled, torso rotating horizontal while legs splay apart.
walker2d 150k  seg4  Forward hopping with asymmetric leg sweep, body rising from lean to upright posture.
walker2d  50k  seg4  Slow single-leg balance with forward lean, poor stability, stumbling recovery.

hopper   200k  seg2  Slow rhythmic hopping with alternating bent-knee stance and extended push-off.
hopper   100k  seg1  Hopper maintains upright posture with moderate knee flexion, executing low-frequency hops.
hopper    40k  seg3  Forward collapse from upright stance without hopping recovery.

halfcheetah 300k seg3  Low crouched gallop with alternating limb cycles, stable rightward locomotion.
halfcheetah 150k seg1  Ground-hugging crawl with alternating leg extension, low stride height.
halfcheetah  50k seg2  Low-slung leftward gallop with alternating leg extension, torso nearly parallel to ground.
```

The descriptions are concrete and physically grounded (joint angles, stride
character, torso orientation), which is what makes them embed cleanly.

## Retrieval demos

Cosine similarity to the query (higher = closer).

**`"stable forward hopping rhythm"`** → top-3
```
1. [+0.70]  walker2d_150k  seg4   Forward hopping with asymmetric leg sweep…
2. [+0.70]  hopper_200k    seg1   Slow quasi-static hopping with minimal ground clearance…
3. [+0.68]  hopper_200k    seg2   Slow rhythmic hopping with alternating bent-knee stance…
```

**`"low ground crawl forward"`** → top-3
```
1. [+0.73]  halfcheetah_150k seg3  Low-ground crawling with asymmetric forelimb cycling…
2. ...     halfcheetah_50k  seg1  Low horizontal belly-crawl with folding limbs…
3. ...     halfcheetah_150k seg1  Ground-hugging crawl with alternating leg extension…
```

**`"falling toppling collapse"`** → top-3
```
1. [+0.65]  walker2d_50k  seg1   Forward toppling fall from vertical stance…
2. ...     walker2d_50k  seg2   Walker2d falls forward uncontrolled…
3. ...     walker2d_150k seg2   Walker2d topples forward uncontrollably from upright to horizontal collapse…
```

**`"fast running gallop"`** filtered to HalfCheetah → top-3
```
1. [+0.62]  halfcheetah_50k  seg2   Low-slung leftward gallop with alternating leg extension…
2. [+0.60]  halfcheetah_300k seg3   Low crouched gallop with alternating limb cycles…
3. [+0.36]  halfcheetah_50k  seg3   Low-slung belly-crawl bound…
```

**`--neighbors-of hopper__s0__step200000__seg2`** (anchor: "Slow rhythmic
hopping with alternating bent-knee stance and extended push-off")
```
1. [+0.83]  walker2d_150k seg4  Forward hopping with asymmetric leg sweep…          (cross-env neighbor)
2. [+0.81]  hopper_200k   seg3  Low-amplitude in-place hop with minimal lift…       (same policy, similar gait)
3. [+0.69]  hopper_100k   seg1  Hopper maintains upright posture, low-freq hops…    (earlier checkpoint, same env)
4. [+0.69]  hopper_100k   seg3  Upright hopper maintains shallow knee flexion…      (earlier checkpoint, same env)
5. [+0.62]  walker2d_50k  seg4  Slow single-leg balance with forward lean…           (analogous balance segment)
```

This last case is the key result for the user story: given a known-good
hopping policy/segment, the index surfaces (a) other gait-similar segments
from the *same* policy, (b) other checkpoints of the same policy, and (c)
*cross-environment* analogues — a hopping pattern in Walker2d shows up as
the top neighbor of a Hopper policy. That cross-env retrieval is what
makes "find policies near this behavior" actually useful.

## Honest limitations

- **PPO at 300k steps is undertrained for these envs.** Many "best"
  checkpoints still topple, so the index is heavy on collapse modes.
  Bumping `total_steps` in `src/policies/catalog.py` to 1M–2M would give
  crisper running gaits; budget is cheap (~5 min/policy at 4.7k fps).
- **Single seed per env.** No noise-floor analysis; segments from a
  different seed of the same env could look quite different.
- **Concise-string embeddings only.** We embed the Haiku-condensed string
  for retrieval and keep the detailed Sonnet description as metadata.
  Embedding the detailed string instead would give finer-grained matches
  but cost more index space and dilute tag-style queries. Worth ablating.
- **VLM consistency.** Sonnet's descriptions are reasonably stable across
  similar segments but not deterministic; two near-identical gait phases
  may get embedded ~0.1 cosine apart due to wording. Quantitative
  consistency check (cosine variance within a single repeated rollout) is
  a natural follow-up.
- **Segment count is sensitive to `penalty_mult`.** Default penalty in
  `src/rollout/segment.py` produced 3-6 segments per ~600-step rollout,
  which is reasonable. On longer rollouts or noisier envs the same
  penalty may over- or under-segment.

## Index state

```
collection:  behaviors
path:        iter2/data/chroma/
count:       41
distance:    cosine
embed model: sentence-transformers/all-MiniLM-L6-v2  (384-d)
```

## Reproduce

```bash
cd iter2
bash setup.sh
source .venv/bin/activate

# .env already contains ANTHROPIC_API_KEY (gitignored)
python -m src.verify                          # deps + Walker2d round-trip
python -m src.policies.train --all            # ~3 min
python pipeline.py --steps 600                # rollout + VLM + index
python search.py "<your query>" -k 5          # retrieval
python search.py --neighbors-of <segment_id>  # "find policies near this one"
python search.py --stats                      # current collection size
```
