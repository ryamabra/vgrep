"""Build a benchmark corpus from Pexels.

Downloads a fixed set of categories into one folder per category, which gives
ground truth for free: a photo's folder is the label. That turns "the scores
look about right" into a measurable precision@k.

Usage:
    export PEXELS_API_KEY=your_key_here
    python tools/fetch_corpus.py                 # default: 15 per category
    python tools/fetch_corpus.py --per 25        # more photos
    python tools/fetch_corpus.py --out ~/corpus  # somewhere else

Rate limit is 200 requests/hour. One request covers up to 80 photos, so the
default run costs ~22 requests -- well inside the limit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.pexels.com/v1/search"

# Toronto for the theme, generic categories so queries actually have to
# discriminate. A corpus of nothing but skylines proves nothing -- the value
# is in "dogs" returning dogs rather than buildings.
CATEGORIES = [
    # Toronto
    "CN Tower Toronto",
    "Toronto skyline",
    "Toronto streetcar",
    "Nathan Phillips Square",
    "Distillery District Toronto",
    "Casa Loma",
    "Toronto Islands",
    "Kensington Market Toronto",
    # Generic, visually distinct
    "plated food restaurant",
    "dog portrait",
    "cat",
    "coffee shop interior",
    "laptop desk workspace",
    "suspension bridge",
    "sports car",
    "mountain landscape",
    "beach sunset",
    "concert crowd",
    "bookshelf library",
    "bicycle city street",
    "snow winter forest",
    "flowers close up",
]


def slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def fetch_page(query: str, per_page: int, key: str) -> list[dict]:
    url = f"{API}?" + urllib.parse.urlencode(
        {"query": query, "per_page": per_page, "orientation": "landscape"}
    )
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": key,
            # Pexels rejects the default "Python-urllib/3.x" agent with a 403.
            # Any ordinary UA string works.
            "User-Agent": "vgrep-corpus-builder/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        import json

        return json.load(resp).get("photos", [])


def download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "vgrep-corpus-builder/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"    failed: {e}", file=sys.stderr)
        return False
    dest.write_bytes(data)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=15, help="photos per category")
    ap.add_argument("--out", default="~/vgrep-corpus", help="output directory")
    args = ap.parse_args()

    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        print("Set PEXELS_API_KEY first:", file=sys.stderr)
        print("  export PEXELS_API_KEY=your_key_here", file=sys.stderr)
        return 1

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()  # Pexels ids, so a photo matching two queries lands once
    rows: list[dict] = []
    total = 0

    for i, query in enumerate(CATEGORIES, 1):
        folder = out / slug(query)
        folder.mkdir(exist_ok=True)
        print(f"[{i}/{len(CATEGORIES)}] {query}")

        try:
            photos = fetch_page(query, args.per * 2, key)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("  rate limited -- stopping. Re-run later to continue.", file=sys.stderr)
                break
            print(f"  error {e.code}: {e.reason}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"  error: {e}", file=sys.stderr)
            continue

        got = 0
        for p in photos:
            if got >= args.per:
                break
            pid = str(p.get("id"))
            if pid in seen:
                continue
            seen.add(pid)

            src = p.get("src", {}).get("large")
            if not src:
                continue

            name = f"{pid}.jpg"
            dest = folder / name
            if dest.exists() or download(src, dest):
                got += 1
                total += 1
                rows.append(
                    {
                        "path": str(dest),
                        "category": query,
                        "pexels_id": pid,
                        "photographer": p.get("photographer", ""),
                        "url": p.get("url", ""),
                    }
                )

        print(f"  {got} photos")
        time.sleep(0.5)  # be polite

    manifest = out / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "category", "pexels_id", "photographer", "url"]
        )
        w.writeheader()
        w.writerows(rows)

    print(f"\n{total} photos across {len(set(r['category'] for r in rows))} categories")
    print(f"  {out}")
    print(f"  {manifest}  (category labels = ground truth)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
