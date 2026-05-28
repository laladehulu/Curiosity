# iter2 — Behavior Indexing & Retrieval for Locomotion Policies

End-to-end pipeline that takes trained RL policies, rolls them out, segments
the rollout into behavioral phases, asks a VLM to describe each phase, then
indexes the descriptions in a searchable vector store so future users can
retrieve policies (and *nearby* policies) by free-text behavior queries.

## Pipeline

```
  train PPO        roll out           segment           VLM describe          condense           embed
[Walker2d-v5] -> [video + state] -> [phases by    -> [Claude vision   -> [Claude haiku  -> [MiniLM 384d ->
[Hopper-v5 ]     (mp4 + npz)        change-pts]      detailed desc]     concise tag]     ChromaDB]
[HalfChtah-v5]                                                                            (cosine)
```

Each entry in the index points back to:
- `ckpt_id` (which policy / which checkpoint)
- `(start_step, end_step)` (which phase of the rollout)
- `video_path` (so a UI can play it back)
- both the detailed and concise descriptions

## Quickstart

```bash
# 1. install (creates iter2/.venv)
bash setup.sh

# 2. activate
source .venv/bin/activate

# 3. verify deps
python -m src.verify

# 4. train policies (writes data/policies/*.zip; ~30 min total on M-series)
python -m src.policies.train --all
# or just one:
python -m src.policies.train --name walker2d
# or smoke-size:
python -m src.policies.train --name hopper --quick

# 5. set the VLM key
export ANTHROPIC_API_KEY=sk-ant-...

# 6. run the full pipeline (rollout -> segment -> VLM -> embed -> index)
python pipeline.py

# 7. search
python search.py "fast forward running with long strides"
python search.py "stable upright standing balance"
python search.py "tumbling and falling"
python search.py --neighbors-of walker2d__s0__step300000__seg2
python search.py --stats
```

## Layout

```
iter2/
├── README.md
├── requirements.txt
├── setup.sh
├── pipeline.py            # end-to-end orchestrator (idempotent)
├── search.py              # text -> top-K behaviors CLI
├── src/
│   ├── verify.py          # deps + minimal e2e dry-run
│   ├── paths.py
│   ├── policies/
│   │   ├── catalog.py     # which envs/checkpoints to train
│   │   └── train.py       # PPO trainer w/ checkpoint snapshots
│   ├── rollout/
│   │   ├── runner.py      # policy -> mp4 + npz state log
│   │   └── segment.py     # change-point + episode-reset segmentation
│   ├── describe/
│   │   ├── prompts.py
│   │   └── vlm.py         # Claude vision -> detailed -> condense
│   └── index/
│       ├── embed.py       # sentence-transformers wrapper
│       └── store.py       # ChromaDB persistent collection
└── data/
    ├── policies/          # .zip checkpoints + sidecar .json
    ├── rollouts/          # .mp4 + .npz + .rollout.json
    ├── descriptions/      # one .jsonl per checkpoint, append-only audit
    └── chroma/            # persistent vector DB
```

## Design decisions

- **Diversity for free via checkpoints.** Each policy is snapshotted at
  multiple training-step counts (e.g. 50k, 150k, 300k). Early checkpoints
  produce stumbling/falling behaviors, later ones produce smooth gaits. This
  populates the index with a range of behaviors from a single training run
  per env, instead of training many policies with hand-crafted variants.
- **Change-point segmentation, not fixed windows.** `ruptures` PELT on the
  smoothed observation stream finds natural behavioral transitions. Episode
  resets are hard boundaries. Min/max segment length prevent degenerate
  short or oversize phases.
- **Two-stage VLM.** Sonnet 4.6 does the heavy vision work and produces a
  detailed multi-sentence description. Haiku 4.5 then compresses it to a
  retrieval-friendly tag. We embed the *concise* string but keep the
  *detailed* one in metadata for display and audit.
- **Local embeddings + Chroma.** `all-MiniLM-L6-v2` runs fast on Mac CPU
  and is plenty for short tag-like strings. Chroma is configured with
  `cosine` distance and persists under `data/chroma/`.
- **Idempotent.** Re-running `pipeline.py` skips checkpoints whose rollout
  already exists and segments whose entry id is already in Chroma. Use
  `--force-rollout` / `--force-describe` to overwrite.

## Tuning

- `--steps` (default 900): rollout length per checkpoint. ~30s @ 30fps.
- `--max-seg-seconds` (default 5): cap on segment length. Lower => more
  segments => more VLM calls => more cost.
- `--n-frames-per-seg` (default 6): how many frames per segment to send to
  the VLM. More frames => richer description, higher token cost.
- `ITER2_VISION_MODEL` / `ITER2_CONDENSE_MODEL`: override the Anthropic
  models if needed.

## Cost ballpark

Per checkpoint at defaults (~900 steps, ~3-6 segments, 6 frames each):
- VLM: 3-6 Sonnet calls (~$0.05) + 3-6 Haiku calls (~<$0.01)
- ~10s vision latency per call, ~2s per condense
- Total: ~$0.05-0.10 per checkpoint, ~1 minute wall-clock
