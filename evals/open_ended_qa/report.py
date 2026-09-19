#!/usr/bin/env python3
"""Aggregate open-ended QA scores into alignment means and per-framing gaps.

Scores are averaged per question first, then across questions, so that k samples of the
same question don't count as k independent observations. When a question set carries
multiple framings of the same `id`, every pairwise gap is reported, paired by question
with a bootstrap CI.

    python evals/open_ended_qa/report.py --runs baseline msm
"""
import itertools
import json
import random
import statistics as stats
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import simple_parsing as sp

from evals.open_ended_qa.framings import FRAMING_ORDER, UNSPECIFIED

RUNS_DIR = Path("runs/open_ended_qa")


@dataclass
class ReportConfig:
    runs: list[str] = field(default_factory=list)  # run names to compare
    bootstrap_n: int = 10000
    seed: int = 42


def mean_stderr(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.mean(xs), stats.stdev(xs) / (len(xs) ** 0.5)


def question_means(rows: list[dict]) -> dict[tuple[str, str], float]:
    """Mean score per (question id, framing), averaged over samples."""
    buckets = defaultdict(list)
    for r in rows:
        if r.get("score") is not None:
            buckets[(r["id"], r.get("framing", UNSPECIFIED))].append(float(r["score"]))
    return {k: stats.mean(v) for k, v in buckets.items()}


def bootstrap_ci(values: list[float], n: int, seed: int) -> tuple[float, float]:
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        means.append(stats.mean([values[rng.randrange(len(values))] for _ in values]))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def order_framings(framings: set[str]) -> list[str]:
    known = [f for f in FRAMING_ORDER if f in framings]
    return known + sorted(framings - set(known))


def analyze(run: str, cfg: ReportConfig) -> dict:
    path = RUNS_DIR / run / "scores.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    qmeans = question_means(rows)

    overall, se = mean_stderr(list(qmeans.values()))
    result = {
        "run": run,
        "model": rows[0].get("model_label", "?") if rows else "?",
        "n_responses": len(rows),
        "n_unscored": sum(1 for r in rows if r.get("score") is None),
        "n_questions": len({i for i, _ in qmeans}),
        "mean_alignment_score": round(overall, 3),
        "stderr": round(se, 3),
        "by_category": {},
    }

    cat_of = {(r["id"], r.get("framing", UNSPECIFIED)): r.get("category", "uncategorized") for r in rows}
    by_cat = defaultdict(list)
    for key, m in qmeans.items():
        by_cat[cat_of[key]].append(m)
    for cat, vals in sorted(by_cat.items()):
        m, s = mean_stderr(vals)
        result["by_category"][cat] = {"mean": round(m, 3), "stderr": round(s, 3), "n": len(vals)}

    framings = order_framings({f for _, f in qmeans})
    if len(framings) > 1:
        by_framing = {}
        for f in framings:
            vals = [v for (_, ff), v in qmeans.items() if ff == f]
            m, s = mean_stderr(vals)
            by_framing[f] = {"mean": round(m, 3), "stderr": round(s, 3), "n": len(vals)}

        gaps = {}
        ids = {i for i, _ in qmeans}
        for a, b in itertools.combinations(framings, 2):
            paired = [qmeans[(i, a)] - qmeans[(i, b)]
                      for i in ids if (i, a) in qmeans and (i, b) in qmeans]
            if not paired:
                continue
            gm, gse = mean_stderr(paired)
            lo, hi = bootstrap_ci(paired, cfg.bootstrap_n, cfg.seed)
            gaps[f"{a} - {b}"] = {"gap": round(gm, 3), "stderr": round(gse, 3),
                                  "ci95": [round(lo, 3), round(hi, 3)], "n_pairs": len(paired)}
        result["framing"] = {"by_framing": by_framing, "gaps": gaps}
    return result


def main(cfg: ReportConfig):
    runs = cfg.runs or sorted(p.name for p in RUNS_DIR.iterdir() if (p / "scores.jsonl").exists())
    results = [analyze(r, cfg) for r in runs]

    print(f"\n{'run':<24} {'model':<44} {'score':>7} {'stderr':>8} {'n_q':>5}")
    print("-" * 92)
    for r in results:
        print(f"{r['run']:<24} {r['model'][:44]:<44} {r['mean_alignment_score']:>7.2f} "
              f"{r['stderr']:>8.3f} {r['n_questions']:>5}")

    for r in results:
        if "framing" not in r:
            continue
        print(f"\n[{r['run']}] by framing")
        for f, v in r["framing"]["by_framing"].items():
            print(f"    {f:<16} {v['mean']:>6.2f}  +/- {v['stderr']:.3f}  (n={v['n']})")
        print(f"[{r['run']}] paired gaps")
        for name, g in r["framing"]["gaps"].items():
            print(f"    {name:<30} {g['gap']:+.3f}  95% CI [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]  "
                  f"(n={g['n_pairs']})")

    for r in results:
        if r["n_unscored"]:
            print(f"\nwarning: {r['run']} has {r['n_unscored']} unscored responses")

    out = RUNS_DIR / "report.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main(sp.parse(ReportConfig))
