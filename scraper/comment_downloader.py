"""Primary scraping path: wraps the `youtube-comment-downloader` library.

No API key required. Works by replaying YouTube's internal comment-pagination
requests, so it's far more stable than driving a real browser — this is the
path `pipeline.py` tries first, falling back to `youtube_scraper.py` (Selenium)
only for videos where this fails (e.g. comments disabled detection edge cases).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Iterator

from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR, SORT_BY_RECENT

_VIDEO_ID_RE = re.compile(
    r"(?:v=|/videos/|embed/|youtu\.be/|/v/|/shorts/)([A-Za-z0-9_-]{11})"
)


@dataclass
class Comment:
    video_id: str
    comment_id: str
    author: str
    text: str
    likes: int
    time_text: str
    reply: bool


def extract_video_id(url_or_id: str) -> str:
    """Accepts a bare 11-char video ID or any common YouTube URL shape."""
    candidate = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate
    match = _VIDEO_ID_RE.search(candidate)
    if not match:
        raise ValueError(f"Could not extract a video ID from: {url_or_id!r}")
    return match.group(1)


def fetch_comments(
    video_id: str,
    limit: int | None = 500,
    sort: str = "popular",
) -> Iterator[Comment]:
    """Yields Comment records for a single video.

    `sort` is "popular" or "recent". `limit=None` fetches all available
    comments (can be a lot for viral videos — use with care).
    """
    sort_key = SORT_BY_POPULAR if sort == "popular" else SORT_BY_RECENT
    downloader = YoutubeCommentDownloader()
    count = 0
    for raw in downloader.get_comments(video_id, sort_by=sort_key):
        if limit is not None and count >= limit:
            return
        yield Comment(
            video_id=video_id,
            comment_id=raw.get("cid", ""),
            author=raw.get("author", ""),
            text=raw.get("text", ""),
            likes=int(raw.get("votes", "0").replace(",", "") or 0)
            if isinstance(raw.get("votes"), str)
            else int(raw.get("votes", 0) or 0),
            time_text=raw.get("time", ""),
            reply=bool(raw.get("reply", False)),
        )
        count += 1


def comment_to_dict(comment: Comment) -> dict:
    return asdict(comment)
