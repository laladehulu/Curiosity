"""Text search over the behavior index.

Usage:
    python search.py "fast forward running"
    python search.py "stable upright balancing" -k 8
    python search.py --neighbors-of walker2d__s0__step300000__seg2
    python search.py --stats
"""
from __future__ import annotations

import argparse
import json

from src.index.store import neighbors_of, query, stats


def _print_hits(hits, header: str):
    print(f"\n{header}")
    if not hits:
        print("  (no results)")
        return
    for rank, h in enumerate(hits, 1):
        print(f"  {rank:>2}. [{h.score:+.3f}]  {h.id}")
        print(f"        {h.concise}")
        print(f"        env={h.metadata.get('env_id')}  "
              f"dur={h.metadata.get('duration_s'):.1f}s  why={h.metadata.get('why')}")
        print(f"        video: {h.metadata.get('video_path')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Free-text behavior query")
    ap.add_argument("-k", "--k", type=int, default=5)
    ap.add_argument("--env", help="Filter by env_id (e.g. Walker2d-v4)")
    ap.add_argument("--neighbors-of", help="Find behaviors near this segment id")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        print(json.dumps(stats(), indent=2))
        return

    if args.neighbors_of:
        hits = neighbors_of(args.neighbors_of, k=args.k)
        _print_hits(hits, f"neighbors of {args.neighbors_of}")
        return

    if not args.text:
        ap.error("provide a query text, or --neighbors-of ID, or --stats")

    where = {"env_id": args.env} if args.env else None
    hits = query(args.text, k=args.k, where=where)
    _print_hits(hits, f"top-{args.k} matches for {args.text!r}")


if __name__ == "__main__":
    main()
