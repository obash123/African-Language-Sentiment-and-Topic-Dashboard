"""CLI orchestrator: video list -> scrape -> deduped CSV in data/raw/.

Usage:
    python -m scraper.pipeline --urls-file videos.txt --out data/raw/comments.csv
    python -m scraper.pipeline --video-ids dQw4w9WgXcQ abc123XYZ_9 --limit 300

Tries comment_downloader.py (fast, no API key) for every video. Videos it
fails on are collected and reported so youtube_scraper.py's Selenium path
can be run on that shortlist separately.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .comment_downloader import extract_video_id, fetch_comments, comment_to_dict

FIELDNAMES = ["video_id", "comment_id", "author", "text", "likes", "time_text", "reply"]


def read_video_ids(args: argparse.Namespace) -> list[str]:
    raw_refs: list[str] = list(args.video_ids or [])
    if args.urls_file:
        path = Path(args.urls_file)
        raw_refs.extend(
            line.strip() for line in path.read_text().splitlines() if line.strip()
        )
    if not raw_refs:
        raise SystemExit("Provide --video-ids and/or --urls-file")
    return [extract_video_id(ref) for ref in raw_refs]


def run(video_ids: list[str], out_path: Path, limit: int | None, sort: str) -> list[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_comment_ids: set[str] = set()
    failed_video_ids: list[str] = []

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for video_id in video_ids:
            try:
                count_before = len(seen_comment_ids)
                for comment in fetch_comments(video_id, limit=limit, sort=sort):
                    key = comment.comment_id or f"{comment.video_id}:{comment.author}:{comment.text[:40]}"
                    if key in seen_comment_ids:
                        continue
                    seen_comment_ids.add(key)
                    writer.writerow(comment_to_dict(comment))
                if len(seen_comment_ids) == count_before:
                    failed_video_ids.append(video_id)
            except Exception as exc:  # network / library errors for this one video
                print(f"[pipeline] failed on {video_id}: {exc}", file=sys.stderr)
                failed_video_ids.append(video_id)

    return failed_video_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-ids", nargs="*", help="Bare video IDs or full URLs")
    parser.add_argument("--urls-file", help="Text file with one video URL/ID per line")
    parser.add_argument("--out", default="data/raw/comments.csv")
    parser.add_argument("--limit", type=int, default=500, help="Max comments per video")
    parser.add_argument("--sort", choices=["popular", "recent"], default="popular")
    args = parser.parse_args()

    video_ids = read_video_ids(args)
    failed = run(video_ids, Path(args.out), args.limit, args.sort)

    print(f"[pipeline] scraped {len(video_ids) - len(failed)}/{len(video_ids)} videos -> {args.out}")
    if failed:
        print(
            "[pipeline] these video IDs got 0 comments via the downloader path "
            f"and may need the Selenium fallback in youtube_scraper.py: {failed}"
        )


if __name__ == "__main__":
    main()
