"""Measure retrieval quality against the corpus manifest.

Every photo's folder is its label, so we can compute real precision@k instead of
eyeballing scores. For each category we run a natural-language query and count
how many of the top k results actually came from that category's folder.

Two things this deliberately does NOT do:

  - Use the folder name as the query. Querying "dog_portrait" would test string
    matching, not semantics. The QUERIES table below maps each category to a
    phrase a person would actually type, which is the thing worth measuring.
  - Reuse the CLI. It calls the library directly so one model load covers all
    queries instead of paying ~2s of startup per search.

Usage:
    python tools/benchmark.py                      # uses ~/vgrep-corpus
    python tools/benchmark.py --corpus ~/other     # elsewhere
    python tools/benchmark.py --k 5                # precision@5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

# Natural phrasing, not folder names. A few are deliberately hard: proper nouns
# ("CN Tower", "Casa Loma") test whether the model knows a named landmark rather
# than just a visual category, which is a different and weaker capability.
QUERIES = {
    "CN Tower Toronto": "the CN Tower",
    "Toronto skyline": "a city skyline at dusk",
    "Toronto streetcar": "a streetcar on a city street",
    "Nathan Phillips Square": "a public square with a fountain",
    "Distillery District Toronto": "a cobblestone street with brick buildings",
    "Casa Loma": "a castle with towers",
    "Toronto Islands": "a view of the city from across the water",
    "Kensington Market Toronto": "a colorful shopfront on a busy street",
    "plated food restaurant": "food on a plate",
    "dog portrait": "a dog",
    "cat": "a cat",
    "coffee shop interior": "inside a coffee shop",
    "laptop desk workspace": "a laptop on a desk",
    "suspension bridge": "a large bridge",
    "sports car": "a car",
    "mountain landscape": "mountains",
    "beach sunset": "a sunset over the ocean",
    "concert crowd": "a crowd at a concert",
    "bookshelf library": "shelves full of books",
    "bicycle city street": "a bicycle",
    "snow winter forest": "snow covered trees",
    "flowers close up": "flowers",
}


def load_manifest(corpus: Path) -> dict[str, str]:
    """path -> category, from the download manifest."""
    manifest = corpus / "manifest.csv"
    if not manifest.exists():
        print(f"No manifest at {manifest}", file=sys.stderr)
        print("Run tools/fetch_corpus.py first.", file=sys.stderr)
        raise SystemExit(1)

    with manifest.open() as f:
        return {row["path"]: row["category"] for row in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="~/vgrep-corpus")
    ap.add_argument("--k", type=int, default=10, help="precision@k")
    args = ap.parse_args()

    corpus = Path(args.corpus).expanduser()
    labels = load_manifest(corpus)

    from vgrep import index as fi
    from vgrep.config import SETTINGS
    from vgrep.db import Db
    from vgrep.encoder import Encoder

    db = Db(SETTINGS.db_path)
    idx = fi.load(SETTINGS.index_path)
    if idx is None or idx.ntotal == 0:
        print("Nothing indexed. Run `vgrep index` first.", file=sys.stderr)
        return 1

    # Warn about anything indexed that isn't part of the labelled corpus --
    # stray files silently depress precision and make the numbers meaningless.
    total_indexed = idx.ntotal
    unlabelled = total_indexed - len(labels)
    if unlabelled > 0:
        print(
            f"Warning: {total_indexed} images indexed but only {len(labels)} labelled.\n"
            f"         {unlabelled} unlabelled images will count as misses.\n"
            f"         Run `vgrep reset -y && vgrep index {args.corpus}` for a clean run.\n",
            file=sys.stderr,
        )

    enc = Encoder()
    print(f"Corpus: {total_indexed} images, {len(set(labels.values()))} categories")
    print(f"Device: {enc.device}   Model: {SETTINGS.model}\n")

    results: list[tuple[str, str, float, float]] = []
    latencies: list[float] = []

    for category, query in QUERIES.items():
        t0 = time.time()
        qvec = enc.encode_text([query])[0]
        hits = fi.search(idx, qvec, args.k)
        latencies.append(time.time() - t0)

        paths = db.paths_for(i for i, _ in hits)
        correct = sum(1 for fid, _ in hits if labels.get(paths.get(fid, "")) == category)
        precision = correct / max(len(hits), 1)
        top_score = hits[0][1] if hits else 0.0

        results.append((category, query, precision, top_score))

    # Report, worst first: the failures are the interesting part.
    results.sort(key=lambda r: r[2])

    width = max(len(c) for c in QUERIES)
    print(f"{'category':<{width}}  {'query':<38}  P@{args.k}   top")
    print("-" * (width + 56))
    for category, query, precision, top in results:
        flag = "" if precision >= 0.8 else ("  <-- weak" if precision >= 0.3 else "  <-- FAIL")
        print(f"{category:<{width}}  {query:<38}  {precision:5.0%}  {top:5.1%}{flag}")

    overall = statistics.mean(r[2] for r in results)
    perfect = sum(1 for r in results if r[2] == 1.0)
    print("-" * (width + 56))
    print(f"mean precision@{args.k}: {overall:.1%}")
    print(f"perfect categories:   {perfect}/{len(results)}")
    print(f"median query latency: {statistics.median(latencies) * 1000:.0f} ms")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
