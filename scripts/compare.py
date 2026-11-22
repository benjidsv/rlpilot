"""Paired comparison of two evaluations on the same scenario suite (CSV files written by training or scripts/eval.py).

    python scripts/compare.py runs/A/eval/0050003968.csv runs/B/eval/0050003968.csv [--tracks]

Both played exactly the same scenarios, so instead of comparing two averages, it counts the scenarios one passes and
the other fails. Pass = survived (clean or survived) for every tier except crashed, where it is completed (anything but
failed). The p-value is an exact two-sided sign test on those discordant scenarios: how surprising the split would be
if A and B were equally good. Scenarios aren't fully independent (neighbouring ones share a corner), so read it as a
guide; --tracks shows whether a difference is spread out or comes from one track.
"""

import argparse
from math import comb
from pathlib import Path

from rlpilot.evaluate import read_results
from rlpilot.starts import KINDS


def passed(r: dict) -> bool:
    return r["outcome"] != "failed" if r["tier"] == "crashed" else r["outcome"] in ("clean", "survived")


def sign_test(a_only: int, b_only: int) -> float:
    n, k = a_only + b_only, min(a_only, b_only)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def table(pairs, key) -> None:
    groups = {}
    for a, b in pairs:
        groups.setdefault(key(a), []).append((passed(a), passed(b)))
    print(f"  {'':28s} {'n':>5s} {'A pass':>7s} {'B pass':>7s} {'A only':>7s} {'B only':>7s} {'p':>7s}")
    for g, v in groups.items():
        a_only, b_only = sum(x and not y for x, y in v), sum(y and not x for x, y in v)
        print(f"  {g:28s} {len(v):5d} {sum(x for x, _ in v) / len(v):7.3f} {sum(y for _, y in v) / len(v):7.3f} "
              f"{a_only:7d} {b_only:7d} {sign_test(a_only, b_only):7.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    p.add_argument("--tracks", action="store_true", help="also break the comparison down by track")
    args = p.parse_args()
    a = {r["id"]: r for r in read_results(args.a)}
    b = {r["id"]: r for r in read_results(args.b)}
    common = [i for i in a if i in b]
    if len(common) < max(len(a), len(b)):
        print(f"note: {len(a)} and {len(b)} scenarios, {len(common)} in common (compared on those)")
    pairs = sorted(((a[i], b[i]) for i in common), key=lambda ab: (ab[0]["split"], KINDS.index(ab[0]["tier"])))
    print(f"A = {args.a}\nB = {args.b}\n")
    table(pairs, lambda r: f"{r['split']} / {r['tier']}")
    if args.tracks:
        for split in sorted({x["split"] for x, _ in pairs}):
            print(f"\n  {split}, by track:")
            table(sorted((ab for ab in pairs if ab[0]["split"] == split), key=lambda ab: ab[0]["track"]),
                  lambda r: r["track"])


if __name__ == "__main__":
    main()
