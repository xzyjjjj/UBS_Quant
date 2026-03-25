from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class GDELTRateLimitError(RuntimeError):
    def __init__(self, message: str, *, retry_after_secs: Optional[int] = None) -> None:
        super().__init__(message)
        self.retry_after_secs = retry_after_secs


@dataclass(frozen=True)
class GDELTDocQuery:
    query: str
    startdatetime: str  # YYYYMMDDHHMMSS
    enddatetime: str  # YYYYMMDDHHMMSS
    mode: str = "ArtList"
    format: str = "json"
    sort: str = "HybridRel"
    maxrecords: int = 250
    startrecord: int = 1


def _http_get_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    req = Request(
        url,
        headers={
            "User-Agent": "UBS_Quant/0.1 (gdelt doc api)",
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
    )
    try:
        with urlopen(req, timeout=timeout) as resp:  # nosec - intended
            raw = resp.read()
            headers = resp.headers
            content_type = headers.get("Content-Type", "")
            content_encoding = headers.get("Content-Encoding", "")
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")[:200]
        except Exception:
            body = ""
        if int(getattr(e, "code", 0) or 0) == 429:
            retry_after = None
            try:
                ra = e.headers.get("Retry-After") if getattr(e, "headers", None) is not None else None
                if ra is not None:
                    retry_after = int(str(ra).strip())
            except Exception:
                retry_after = None
            raise GDELTRateLimitError(f"GDELT HTTP 429: {body}", retry_after_secs=retry_after) from e
        raise RuntimeError(f"GDELT HTTP {e.code}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"GDELT connection error: {e}") from e
    except (TimeoutError, socket.timeout, ConnectionError, OSError) as e:
        raise RuntimeError(f"GDELT connection error: {e}") from e

    import json
    import gzip

    if str(content_encoding).lower().strip() == "gzip":
        try:
            raw = gzip.decompress(raw)
        except Exception:
            # If decompression fails, keep raw as-is and let JSON decode handle the error.
            pass

    text = raw.decode("utf-8", errors="strict") if raw else ""
    try:
        return json.loads(text)
    except Exception:
        # Provide a helpful error: often the API returns HTML/text when rate limited or blocked.
        preview = (text[:200] if isinstance(text, str) else "") or (raw[:200].decode("utf-8", errors="ignore"))
        raise RuntimeError(
            "GDELT response is not valid JSON. "
            f"content_type={content_type!r} content_encoding={content_encoding!r} preview={preview!r}"
        )


def build_gdelt_doc_url(q: GDELTDocQuery, *, base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc") -> str:
    base = str(base_url).rstrip("?")
    params = {
        "query": q.query,
        "mode": q.mode,
        "format": q.format,
        "startdatetime": q.startdatetime,
        "enddatetime": q.enddatetime,
        "sort": q.sort,
        "maxrecords": str(int(q.maxrecords)),
        "startrecord": str(int(q.startrecord)),
    }
    return f"{base}?{urlencode(params)}"


def fetch_gdelt_articles_page(
    q: GDELTDocQuery,
    *,
    base_url: str = "https://api.gdeltproject.org/api/v2/doc/doc",
    timeout: int = 30,
) -> dict[str, Any]:
    url = build_gdelt_doc_url(q, base_url=base_url)
    return _http_get_json(url, timeout=timeout)


def parse_gdelt_seendate(value: str) -> Optional[datetime]:
    if not value:
        return None
    value = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            continue
    return None
