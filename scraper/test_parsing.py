"""Offline sanity checks for scraper parsing logic — no network calls.

Run with: python -m scraper.test_parsing
"""
from .comment_downloader import extract_video_id


def test_extract_video_id():
    cases = {
        "dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s": "dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ": "dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ": "dQw4w9WgXcQ",
    }
    for given, expected in cases.items():
        got = extract_video_id(given)
        assert got == expected, f"{given!r} -> {got!r}, expected {expected!r}"

    try:
        extract_video_id("not a url")
        raise AssertionError("expected ValueError for invalid input")
    except ValueError:
        pass


if __name__ == "__main__":
    test_extract_video_id()
    print("scraper.test_parsing: OK")
