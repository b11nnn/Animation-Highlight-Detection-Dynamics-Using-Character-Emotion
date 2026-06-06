from __future__ import annotations

import os
import re
import time
import uuid
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def extract_video_id(youtube_url: str) -> str:
    parsed = urlparse(youtube_url)
    if "youtube.com" in parsed.netloc:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.strip("/")
    raise ValueError("YouTube video_id를 추출하지 못했습니다.")


def make_chrome_driver(chrome_binary: str | None = None) -> webdriver.Chrome:
    os.environ["SE_SKIP_DRIVER_IN_PATH"] = "true"
    options = Options()
    binary = chrome_binary or os.environ.get("CHROME_BIN")
    if binary:
        options.binary_location = binary
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--lang=en-US")
    options.add_argument(f"--user-data-dir=/tmp/chrome-profile-{uuid.uuid4()}")
    return webdriver.Chrome(options=options)


def get_heatmap_svg_path_with_selenium(
    youtube_url: str,
    wait_sec: int = 10,
    chrome_binary: str | None = None,
) -> str | None:
    driver = make_chrome_driver(chrome_binary)
    try:
        driver.get(youtube_url)
        time.sleep(wait_sec)
        try:
            video = driver.find_element(By.TAG_NAME, "video")
            driver.execute_script("arguments[0].muted = true;", video)
            driver.execute_script("arguments[0].play();", video)
            time.sleep(2)
        except Exception:
            pass

        soup = BeautifulSoup(driver.page_source, "html.parser")
        path = soup.find("path", {"class": "ytp-modern-heat-map"})
        if path is None:
            return None
        return path.get("d") or None
    finally:
        driver.quit()


def svg_path_d_to_second_level_replay_df(d: str, duration_sec: float) -> pd.DataFrame:
    pairs = re.findall(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", d)
    if not pairs:
        raise ValueError("SVG path d에서 좌표를 찾지 못했습니다.")

    coord_df = pd.DataFrame([(float(x), float(y)) for x, y in pairs], columns=["x", "y"])
    coord_df = coord_df.query("0 <= x <= 1000")
    if coord_df.empty:
        raise ValueError("유효한 heatmap 좌표가 없습니다.")

    coord_df = (
        coord_df.groupby("x", as_index=False)["y"]
        .min()
        .sort_values("x")
    )
    coord_df["second_float"] = (
        (coord_df["x"] - coord_df["x"].min())
        / (coord_df["x"].max() - coord_df["x"].min())
        * duration_sec
    )

    y_min, y_max = coord_df["y"].min(), coord_df["y"].max()
    coord_df["youtube_replay_score"] = (
        0.0 if y_max == y_min else 1 - ((coord_df["y"] - y_min) / (y_max - y_min))
    )

    seconds = np.arange(0, int(np.ceil(duration_sec)) + 1)
    return pd.DataFrame({
        "second": seconds,
        "youtube_replay_score": np.interp(
            seconds, coord_df["second_float"], coord_df["youtube_replay_score"]
        ),
    })
