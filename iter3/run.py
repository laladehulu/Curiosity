"""iter3 main driver.

Three strategies, same loop body, different selection:
  --strategy archive   k-NN dominance archive over VLM-MiniLM embeddings
  --strategy argmax    Eureka baseline: keep best, mutate best
  --strategy aurora    same as archive but uses AURORA-style AE embedding
                       (AE refit every --aurora-refit-every candidates)

Usage:
  python run.py --strategy archive --budget 20 --seed 0
  python run.py --strategy argmax  --budget 20 --seed 0
  python run.py --strategy aurora  --budget 20 --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src import paths
from src.archive.argmax import ArgmaxArchive
from src.archive.entry import Entry
from src.archive.knn import KNNArchive
from src.describe.vlm import describe_segment
from src.embed_aurora.autoencoder import LATENT_DIM
from src.embed_aurora.pool import POOLED_DIM, pool_trajectory
from src.embed_aurora.train import fit_autoencoder
from src.embed_vlm.embed import embed as embed_vlm_text
from src.eval_grid.descriptor import descriptor as ref_descriptor
from src.llm.generate import GenResult, cold_generate, mutate
from src.rl.eval import rollout_eval
from src.rl.reward_template import CompiledReward, RewardCodeError
from src.rl.train_short import train_short

ENV_ID = "Hopper-v5"
DEFAULT_PPO_STEPS = 30_000
DEFAULT_ROLLOUT_STEPS = 600
DEFAULT_FPS = 30
TAU_CALIBRATE_AT = 5      # after this many entries, set tau_novel from data
AE_REFIT_EVERY = 5        # aurora strategy: refit AE every N CANDIDATES SEEN
AURORA_BOOTSTRAP_TAU = 0.15  # looser bootstrap so the archive grows enough to fit the AE
AURORA_BOOTSTRAP_N = 5    # first N candidates are force-inserted (no dominance), then first AE refit


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class RunConfig:
    strategy: str         # archive | argmax | aurora
    budget: int
    seed: int
    ppo_steps: int
    rollout_steps: int
    fps: int
    aurora_refit_every: int
    n_vlm_frames: int
    cold_start_prob: float
    timestamp: str

    @property
    def run_id(self) -> str:
        return f"{self.strategy}__seed{self.seed}__b{self.budget}__t{self.timestamp}"


def make_run_config(args) -> RunConfig:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    if args.run_id_ts:
        ts = args.run_id_ts
    return RunConfig(
        strategy=args.strategy,
        budget=args.budget,
        seed=args.seed,
        ppo_steps=args.ppo_steps,
        rollout_steps=args.rollout_steps,
        fps=args.fps,
        aurora_refit_every=args.aurora_refit_every,
        n_vlm_frames=args.n_vlm_frames,
        cold_start_prob=args.cold_start_prob,
        timestamp=ts,
    )


# ---- per-candidate work -----------------------------------------------------

def write_reward(cfg: RunConfig, idx: int, code: str, gen_kind: str,
                 parent_id: str | None) -> Path:
    dir_ = paths.run_dir(paths.REWARDS, cfg.run_id)
    p = dir_ / f"reward_{idx:03d}.py"
    p.write_text(code)
    meta = {
        "iteration": idx,
        "prompt_kind": gen_kind,
        "parent_id": parent_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (dir_ / f"reward_{idx:03d}.meta.json").write_text(json.dumps(meta, indent=2))
    return p


def evaluate_candidate(cfg: RunConfig, idx: int, code: str) -> tuple[CompiledReward, dict]:
    """Compile, train, rollout. Returns (compiled, artifact_dict).

    artifact_dict = {
        ckpt_path, video_path, state_path,
        fitness, n_steps, n_episodes, state_npz (loaded),
    }
    """
    try:
        compiled = CompiledReward.from_code(code)
    except RewardCodeError as e:
        raise

    ckpt_path = paths.run_dir(paths.POLICIES, cfg.run_id) / f"policy_{idx:03d}.zip"
    video_path = paths.run_dir(paths.ROLLOUTS, cfg.run_id) / f"rollout_{idx:03d}.mp4"
    state_path = paths.run_dir(paths.ROLLOUTS, cfg.run_id) / f"rollout_{idx:03d}.npz"

    train_short(compiled, ENV_ID, ckpt_path,
                total_timesteps=cfg.ppo_steps, seed=cfg.seed)
    art = rollout_eval(ckpt_path, ENV_ID, video_path, state_path,
                       n_steps=cfg.rollout_steps, fps=cfg.fps, seed=cfg.seed)
    state_npz = dict(np.load(state_path))
    return compiled, {
        "ckpt_path": str(ckpt_path),
        "video_path": str(video_path),
        "state_path": str(state_path),
        "fitness": art.fitness,
        "n_steps": art.n_steps,
        "n_episodes": art.n_episodes,
        "state_npz": state_npz,
    }


def vlm_describe(cfg: RunConfig, video_path: Path) -> tuple[str, str]:
    """Returns (detailed, concise)."""
    d = describe_segment(
        Path(video_path), start_frame=0, end_frame=cfg.rollout_steps,
        env_id=ENV_ID, fps=cfg.fps, n_frames=cfg.n_vlm_frames,
    )
    return d.detailed, d.concise


# ---- archive selection helpers ---------------------------------------------

def _short_id(entry_id: str) -> str:
    return entry_id.rsplit("_", 1)[-1]  # e.g. "020"


def pick_parent_and_prompt(strategy: str, archive, cold_start_prob: float, rng) -> GenResult:
    """Decide cold-gen vs mutate, build the appropriate prompt + call LLM."""
    # Cold start if archive empty, or with small probability
    n = len(archive.entries)
    if n == 0 or rng.random() < cold_start_prob:
        return cold_generate()
    parent = archive.sample_parent()
    if parent is None:
        return cold_generate()

    parent_code = Path(parent.reward_path).read_text()
    parent_desc = parent.description or "(no description)"
    if strategy == "argmax":
        return mutate(parent_code, parent_desc, parent.fitness, parent.id, neighbors=None)
    # archive/aurora: include neighbors
    neighbors_raw = archive.k_nearest(parent, k=3)
    neighbors = [
        (_short_id(e.id), e.fitness, e.description or "(no desc)")
        for e in neighbors_raw if e.id != parent.id
    ]
    return mutate(parent_code, parent_desc, parent.fitness, parent.id, neighbors=neighbors)


# ---- aurora refit ----------------------------------------------------------

def aurora_refit_and_reembed(archive: KNNArchive, seed: int, verbose: bool = False):
    """Train AE on the full corpus of all rollouts seen so far (not just
    archive entries — many were dominated and dropped). Re-embed all
    archive entries with the fresh AE. Rebuild archive under new latent.
    """
    if not archive.entries:
        return None
    # Train on ALL rollouts on disk for this run, not just archive entries.
    rollout_dir = Path(archive.entries[0].state_path).parent
    all_state_paths = sorted(rollout_dir.glob("rollout_*.npz"))
    feats: list[np.ndarray] = []
    for p in all_state_paths:
        feats.append(pool_trajectory(dict(np.load(p))))
    if not feats:
        return None
    train_arr = np.stack(feats)
    model = fit_autoencoder(train_arr, epochs=50, lr=1e-3, seed=seed, verbose=verbose)

    # Re-embed archive entries
    arc_feats = np.stack([pool_trajectory(dict(np.load(e.state_path)))
                          for e in archive.entries])
    new_embs = model.embed(arc_feats)
    for e, ne in zip(archive.entries, new_embs):
        e.embed_aurora = ne.astype(np.float32)
    # rebuild archive
    old_entries = list(archive.entries)
    archive.entries = []
    archive.tau_novel = AURORA_BOOTSTRAP_TAU
    for e in old_entries:
        archive.insert(e)
    if len(archive.entries) >= 3:    # was 5; relaxed so calibration fires earlier
        archive.calibrate_tau()
    return model


# ---- log writer ------------------------------------------------------------

def log_candidate(cfg: RunConfig, line: dict) -> None:
    log_path = paths.LOGS / f"{cfg.run_id}.jsonl"
    with log_path.open("a") as fh:
        fh.write(json.dumps(line) + "\n")


# ---- main loop -------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["archive", "argmax", "aurora"], required=True)
    ap.add_argument("--budget", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ppo-steps", type=int, default=DEFAULT_PPO_STEPS)
    ap.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--aurora-refit-every", type=int, default=AE_REFIT_EVERY)
    ap.add_argument("--n-vlm-frames", type=int, default=6)
    ap.add_argument("--cold-start-prob", type=float, default=0.0)
    ap.add_argument("--run-id-ts", default=None,
                    help="Override timestamp portion of run_id (for resume).")
    args = ap.parse_args()

    seed_everything(args.seed)
    cfg = make_run_config(args)
    rng = random.Random(args.seed * 7919 + 13)

    # Archive
    archive_path = paths.ARCHIVES / f"{cfg.run_id}.json"
    if cfg.strategy == "argmax":
        archive = ArgmaxArchive.load(archive_path) if archive_path.exists() else ArgmaxArchive()
    elif cfg.strategy == "archive":
        archive = (KNNArchive.load(archive_path, embed_key="vlm", rng_seed=cfg.seed)
                   if archive_path.exists()
                   else KNNArchive(embed_key="vlm", rng_seed=cfg.seed))
    elif cfg.strategy == "aurora":
        archive = (KNNArchive.load(archive_path, embed_key="aurora", rng_seed=cfg.seed)
                   if archive_path.exists()
                   else KNNArchive(embed_key="aurora", rng_seed=cfg.seed,
                                   tau_novel=AURORA_BOOTSTRAP_TAU))
    else:
        raise SystemExit(cfg.strategy)

    print(f"[run] run_id={cfg.run_id}")
    print(f"[run] resume from {len(archive.entries)} entries; target {cfg.budget}")

    aurora_model = None
    if cfg.strategy == "aurora" and archive.entries:
        # ensure embeddings consistent on resume
        aurora_model = aurora_refit_and_reembed(archive, seed=cfg.seed)

    start_idx = len(archive.entries)
    for idx in range(start_idx, cfg.budget):
        t0 = time.time()
        print(f"\n=== iter {idx}/{cfg.budget} [{cfg.strategy}] ===")

        # 1. generate
        try:
            gen = pick_parent_and_prompt(cfg.strategy, archive, cfg.cold_start_prob, rng)
        except Exception:
            traceback.print_exc()
            print("[skip] LLM gen failed; retrying with cold-gen")
            gen = cold_generate()

        # write reward to disk early (audit trail)
        reward_path = write_reward(cfg, idx, gen.code, gen.prompt_kind, gen.parent_id)

        # 2. compile + train + rollout
        try:
            compiled, art = evaluate_candidate(cfg, idx, gen.code)
        except RewardCodeError as e:
            print(f"[skip] reward rejected: {e}")
            log_candidate(cfg, {
                "iteration": idx, "id": f"r{idx:03d}", "skipped": True,
                "reason": str(e), "prompt_kind": gen.prompt_kind,
                "parent_id": gen.parent_id,
            })
            continue
        except Exception as e:
            traceback.print_exc()
            log_candidate(cfg, {
                "iteration": idx, "id": f"r{idx:03d}", "skipped": True,
                "reason": f"eval_error: {e!r}", "prompt_kind": gen.prompt_kind,
                "parent_id": gen.parent_id,
            })
            continue

        # 3. VLM describe (always — used by all strategies for downstream eval)
        try:
            detailed, concise = vlm_describe(cfg, Path(art["video_path"]))
        except Exception as e:
            print(f"[warn] VLM failed ({e}); using fallback description")
            detailed = f"VLM call failed: {e}"
            concise = f"(VLM failed) iter {idx} {cfg.strategy}"

        # 4. embeddings
        embed_vlm = np.asarray(embed_vlm_text([concise])[0], dtype=np.float32)
        # ref grid
        ref_coord = ref_descriptor(art["state_npz"], fps=cfg.fps)

        embed_aurora = None
        if cfg.strategy == "aurora":
            pooled = pool_trajectory(art["state_npz"])
            if aurora_model is None:
                # bootstrap: just use raw pooled feature (zero-padded/truncated to latent)
                z = pooled[:LATENT_DIM].astype(np.float32)
                if z.shape[0] < LATENT_DIM:
                    z = np.pad(z, (0, LATENT_DIM - z.shape[0]))
                embed_aurora = z
            else:
                embed_aurora = aurora_model.embed(pooled).astype(np.float32)

        # 5. build entry + insert
        entry_id = f"r{idx:03d}"
        entry = Entry(
            id=entry_id,
            iteration=idx,
            parent_id=gen.parent_id,
            reward_path=str(reward_path),
            fitness=art["fitness"],
            description=concise,
            ckpt_path=art["ckpt_path"],
            video_path=art["video_path"],
            state_path=art["state_path"],
            embed_vlm=embed_vlm,
            embed_aurora=embed_aurora,
            ref_coord=ref_coord,
            prompt_kind=gen.prompt_kind,
        )
        # AURORA strategy: bootstrap phase force-inserts the first N candidates
        # to seed the AE training set with enough diverse rollouts. Otherwise
        # the bootstrap (raw-pooled) embeddings cluster too tightly and every
        # subsequent candidate is dominated before the AE can ever train.
        if cfg.strategy == "aurora" and idx < AURORA_BOOTSTRAP_N:
            archive.entries.append(entry)
            entry.insert_status = "bootstrap"
            status = "bootstrap"
        else:
            status = archive.insert(entry)

        # tau calibration once after warmup (knn archives only)
        if isinstance(archive, KNNArchive) and len(archive.entries) == TAU_CALIBRATE_AT:
            archive.calibrate_tau()
            print(f"[cal] tau_novel calibrated to {archive.tau_novel:.3f}")

        # AURORA refit — based on CANDIDATES SEEN, not archive size. Fires
        # at the end of the bootstrap phase and every refit_every after.
        refit_now = (
            cfg.strategy == "aurora"
            and (
                (idx + 1) == AURORA_BOOTSTRAP_N
                or ((idx + 1) > AURORA_BOOTSTRAP_N
                    and (idx + 1 - AURORA_BOOTSTRAP_N) % cfg.aurora_refit_every == 0)
            )
        )
        if refit_now:
            print(f"[ae] refitting on {len(archive.entries)} archived entries "
                  f"(seen {idx+1} candidates)...")
            if len(archive.entries) >= 2:
                aurora_model = aurora_refit_and_reembed(archive, seed=cfg.seed)
                print(f"[ae] refit done; archive now has {len(archive.entries)} entries "
                      f"(tau_novel={archive.tau_novel:.3f})")
            else:
                print(f"[ae] skip refit — only {len(archive.entries)} archived entry")

        # 6. persist + log
        archive.save(archive_path)
        dt = time.time() - t0
        line = {
            "iteration": idx,
            "id": entry_id,
            "strategy": cfg.strategy,
            "prompt_kind": gen.prompt_kind,
            "parent_id": gen.parent_id,
            "fitness": float(art["fitness"]),
            "insert_status": status,
            "concise": concise,
            "ref_coord": list(ref_coord),
            "wall_clock_s": dt,
            "n_archive_entries": len(archive.entries),
        }
        log_candidate(cfg, line)
        best = max(e.fitness for e in archive.entries)
        print(f"[iter {idx}] fit={art['fitness']:.2f}  status={status}  "
              f"archive={len(archive.entries)}  best={best:.2f}  "
              f"  desc={concise!r}  ({dt:.1f}s)")

    print(f"\n[done] run_id={cfg.run_id}")
    print(f"       final archive: {archive.stats()}")


if __name__ == "__main__":
    main()
