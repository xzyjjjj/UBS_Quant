#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

try:
    from quant_research.data.sources.gdelt import (
        GDELTDocQuery,
        GDELTRateLimitError,
        fetch_gdelt_articles_page,
        parse_gdelt_seendate,
    )
except ImportError:  # allow direct execution without module context
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[3]))
    from quant_research.data.sources.gdelt import (
        GDELTDocQuery,
        GDELTRateLimitError,
        fetch_gdelt_articles_page,
        parse_gdelt_seendate,
    )


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch GDELT news tone and build a daily sentiment score (0-1).")
    p.add_argument("--start", default="2018-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    p.add_argument(
        "--query",
        default='(JPM OR "JPMorgan" OR "JPMorgan Chase")',
        help="GDELT query string",
    )
    p.add_argument("--max-articles", type=int, default=5000, help="Max articles to fetch (cap for pagination)")
    p.add_argument("--page-size", type=int, default=250, help="GDELT maxrecords per request (<=250 recommended)")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds")
    p.add_argument(
        "--gdelt-base-url",
        default="https://api.gdeltproject.org/api/v2/doc/doc",
        help="GDELT DOC endpoint base URL (try http://... if TLS handshake is unstable)",
    )
    p.add_argument(
        "--fallback-http",
        action="store_true",
        help="On repeated TLS/connection errors, retry using http://api.gdeltproject.org/...",
    )
    p.add_argument(
        "--min-interval-secs",
        type=float,
        default=5.2,
        help="Minimum seconds between GDELT requests (default: 5.2, per GDELT rate limit guidance)",
    )
    p.add_argument("--max-retries", type=int, default=6, help="Max retries on HTTP 429 per page")
    p.add_argument("--clip", type=float, default=10.0, help="Clip tone to [-clip, clip] before mapping")
    p.add_argument("--k", type=float, default=2.0, help="Sigmoid scale: score = 1/(1+exp(-(tone/k)))")
    p.add_argument(
        "--out",
        default=None,
        help="Output directory (default: <project_root>/output/news)",
    )
    return p.parse_args()


def _to_dt_yyyymmddhhmmss(d: str, *, end_of_day: bool) -> str:
    ts = datetime.strptime(d, "%Y-%m-%d")
    if end_of_day:
        ts = ts.replace(hour=23, minute=59, second=59)
    else:
        ts = ts.replace(hour=0, minute=0, second=0)
    return ts.strftime("%Y%m%d%H%M%S")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _backoff_seconds(base: float, attempt: int, *, cap: float = 60.0) -> float:
    a = max(int(attempt), 1)
    b = max(float(base), 0.0)
    wait = b * (1.6 ** float(a - 1))
    return min(wait, float(cap))


def main() -> int:
    project_root = Path(__file__).resolve().parents[4]
    _load_dotenv(project_root / ".env")

    args = _parse_args()
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (project_root / "output" / "news")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    startdt = _to_dt_yyyymmddhhmmss(str(args.start), end_of_day=False)
    enddt = _to_dt_yyyymmddhhmmss(str(args.end), end_of_day=True)

    page_size = int(args.page_size)
    page_size = max(1, min(page_size, 250))

    max_articles = max(int(args.max_articles), 0)
    clip = float(args.clip)
    k = float(args.k)
    if k <= 0:
        raise ValueError("--k must be > 0")
    if clip <= 0:
        raise ValueError("--clip must be > 0")

    by_day_tones: dict[str, list[float]] = defaultdict(list)

    query = str(args.query).strip()
    # GDELT requires OR clauses to be grouped in parentheses; auto-wrap the full query when needed.
    if " OR " in query and "(" not in query:
        query = f"({query})"

    fetched = 0
    startrecord = 1
    min_interval = max(float(args.min_interval_secs), 0.0)
    max_retries = max(int(args.max_retries), 0)
    last_request_ts: float | None = None
    gdelt_base_url = str(args.gdelt_base_url).strip()
    http_fallback_used = False

    print("[INFO] GDELT news sentiment pipeline")
    print(f"[INFO] query={query!r}")
    print(f"[INFO] start={args.start} end={args.end} page_size={page_size} max_articles={max_articles or 'unlimited'}")
    print(f"[INFO] gdelt_base_url={gdelt_base_url!r}")
    print(f"[INFO] min_interval_secs={min_interval:.1f} max_retries={max_retries} clip={clip} k={k}")
    while True:
        if max_articles and fetched >= max_articles:
            break

        if last_request_ts is not None and min_interval > 0:
            elapsed = time.time() - last_request_ts
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

        q = GDELTDocQuery(
            query=query,
            startdatetime=startdt,
            enddatetime=enddt,
            maxrecords=page_size,
            startrecord=startrecord,
        )
        attempt = 0
        conn_error_streak = 0
        while True:
            try:
                last_request_ts = time.time()
                payload = fetch_gdelt_articles_page(q, base_url=gdelt_base_url, timeout=int(args.timeout))
                break
            except GDELTRateLimitError as e:
                attempt += 1
                if attempt > max_retries:
                    raise
                wait_hint = float(getattr(e, "retry_after_secs", None) or 5.0)
                wait = _backoff_seconds(max(wait_hint, min_interval, 1.0), attempt, cap=120.0)
                print(f"[WARN] GDELT rate limited (429). Sleep {wait:.1f}s then retry (attempt {attempt}/{max_retries}).")
                time.sleep(wait)
            except RuntimeError as e:
                # GDELT sometimes returns HTML/text (non-JSON) transiently; retry with backoff.
                if "OR'd terms must be surrounded by" in str(e):
                    raise RuntimeError(
                        "GDELT query syntax error: queries containing OR terms must be grouped in parentheses. "
                        f"Query used: {query!r}"
                    ) from e

                if "GDELT connection error" in str(e):
                    conn_error_streak += 1
                    if (
                        bool(args.fallback_http)
                        and (not http_fallback_used)
                        and gdelt_base_url.startswith("https://api.gdeltproject.org/")
                        and conn_error_streak >= 2
                    ):
                        gdelt_base_url = gdelt_base_url.replace("https://", "http://", 1)
                        http_fallback_used = True
                        print(f"[WARN] Switch GDELT base URL to HTTP due to repeated connection errors: {gdelt_base_url!r}")

                attempt += 1
                if attempt > max_retries:
                    raise
                wait = _backoff_seconds(max(min_interval, 8.0), attempt, cap=120.0)
                print(f"[WARN] GDELT transient error. Sleep {wait:.1f}s then retry (attempt {attempt}/{max_retries}).")
                print(f"[WARN] {type(e).__name__}: {e}")
                time.sleep(wait)
        articles = payload.get("articles") or []
        if (not isinstance(articles, list) or not articles) and any(k in payload for k in ("message", "status", "error")):
            msg = payload.get("message") or payload.get("error") or payload.get("status")
            raise RuntimeError(f"GDELT returned no articles with error message: {msg!r}")
        if not isinstance(articles, list) or not articles:
            break

        # Note: fetched_total updates after we parse tones.
        print(f"[INFO] page startrecord={startrecord} articles={len(articles)} fetched_total_before={fetched}")

        for a in articles:
            if max_articles and fetched >= max_articles:
                break
            if not isinstance(a, dict):
                continue
            seen = parse_gdelt_seendate(str(a.get("seendate", "")))
            if seen is None:
                continue
            tone_raw = a.get("tone", None)
            try:
                tone = float(tone_raw)
            except Exception:
                continue
            by_day_tones[seen.date().isoformat()].append(tone)
            fetched += 1

        startrecord += page_size
        print(f"[INFO] page done startrecord={startrecord} fetched_total_after={fetched}")

    out_path = out_dir / "sentiment_daily.csv"
    lines = ["date,gdelt_tone_mean,gdelt_articles,news_sent_01"]
    for day in sorted(by_day_tones.keys()):
        tones = by_day_tones[day]
        if not tones:
            continue
        mean = sum(tones) / float(len(tones))
        mean = max(-clip, min(clip, mean))
        score = _sigmoid(mean / k)
        lines.append(f"{day},{mean:.6f},{len(tones)},{score:.6f}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Articles used: {fetched}")
    if fetched == 0:
        print("[WARN] No articles were parsed. Common causes: still rate-limited, network instability, or query too strict.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
