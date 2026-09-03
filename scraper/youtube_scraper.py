"""Selenium fallback scraper.

`comment_downloader.py` covers the vast majority of videos, but it depends on
YouTube's internal continuation-token API shape, which occasionally changes
or misbehaves for specific videos (e.g. members-only comment sections,
certain live-chat replays). This module drives a real browser, scrolls the
comments panel, and extracts comments from the rendered DOM as a fallback.

Not invoked automatically at import time — call `scrape_video` explicitly.
Requires Chrome installed locally; webdriver-manager fetches a matching
chromedriver on first run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

COMMENT_RENDERER = "ytd-comment-thread-renderer"
COMMENT_TEXT_SELECTOR = "#content-text"
AUTHOR_SELECTOR = "#author-text span"
LIKES_SELECTOR = "#vote-count-middle"


@dataclass
class ScrapedComment:
    video_id: str
    author: str
    text: str
    likes: str


def _build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,2000")
    options.add_argument("--mute-audio")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)


def scrape_video(
    video_id: str,
    max_comments: int = 200,
    scroll_pause_seconds: float = 1.5,
    max_scrolls: int = 40,
    headless: bool = True,
) -> list[ScrapedComment]:
    """Loads the video page, scrolls to load comments, and extracts them.

    This is a real browser-automation routine — it is intentionally never
    called at import time or by any other module in this pipeline. It's here
    so the fallback path exists and is ready to run when someone chooses to.
    """
    driver = _build_driver(headless=headless)
    comments: list[ScrapedComment] = []
    try:
        driver.get(f"https://www.youtube.com/watch?v={video_id}")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight * 0.4);")
        time.sleep(scroll_pause_seconds)

        seen_count = 0
        for _ in range(max_scrolls):
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(scroll_pause_seconds)
            renderers = driver.find_elements(By.TAG_NAME, COMMENT_RENDERER)
            if len(renderers) >= max_comments:
                break
            if len(renderers) == seen_count:
                # No new comments loaded on the last scroll; stop early.
                break
            seen_count = len(renderers)

        for renderer in driver.find_elements(By.TAG_NAME, COMMENT_RENDERER)[:max_comments]:
            try:
                text = renderer.find_element(By.CSS_SELECTOR, COMMENT_TEXT_SELECTOR).text
                author = renderer.find_element(By.CSS_SELECTOR, AUTHOR_SELECTOR).text
                try:
                    likes = renderer.find_element(By.CSS_SELECTOR, LIKES_SELECTOR).text
                except Exception:
                    likes = "0"
            except Exception:
                continue
            comments.append(ScrapedComment(video_id=video_id, author=author, text=text, likes=likes))
    finally:
        driver.quit()
    return comments
