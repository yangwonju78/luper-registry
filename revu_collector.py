"""Stateless REVU review collection endpoints.

Authentication material is accepted only for the lifetime of a request.  It is
never written to the registry database or local files and is never logged.
"""

from __future__ import annotations

import hmac
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from flask import Blueprint, jsonify, request


revu_bp = Blueprint("revu", __name__, url_prefix="/api/revu")

_BAEMIN_HOST = "self-api.baemin.com"
_ALLOWED_HEADERS = {
    "accept",
    "accept-language",
    "authorization",
    "cookie",
    "origin",
    "referer",
    "service-channel",
    "user-agent",
    "x-requested-with",
}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_no_redirect(req: Request, timeout: float):
    # Never forward captured authorization/cookies to a redirect destination.
    return build_opener(_NoRedirect).open(req, timeout=timeout)


def _collector_key() -> str:
    # ADMIN_KEY keeps the first deployment compatible with the existing LUPER
    # Render service. REVU_COLLECTOR_KEY can replace it later for key rotation.
    return os.getenv("REVU_COLLECTOR_KEY") or os.getenv("ADMIN_KEY", "")


def _authorized() -> bool:
    expected = _collector_key()
    supplied = request.headers.get("X-Revu-Key", "")
    return bool(expected and expected != "change-me" and supplied and hmac.compare_digest(expected, supplied))


def _safe_headers(raw) -> dict[str, str]:
    if isinstance(raw, list):
        raw = {str(item.get("name", "")): item.get("value", "") for item in raw if isinstance(item, dict)}
    if not isinstance(raw, dict):
        return {}
    clean = {}
    for name, value in raw.items():
        normalized = str(name).strip().lower()
        if normalized in _ALLOWED_HEADERS and isinstance(value, (str, int, float)):
            clean[normalized] = str(value)
    clean.setdefault("accept", "application/json, text/plain, */*")
    return clean


def _baemin_url(raw_url: str, offset: int, limit: int) -> str:
    parts = urlsplit(raw_url)
    if parts.scheme != "https" or parts.hostname != _BAEMIN_HOST:
        raise ValueError("배민 공식 API 주소만 사용할 수 있습니다.")
    if not parts.path.startswith("/v1/review/shops/") or not parts.path.endswith("/reviews"):
        raise ValueError("배민 리뷰 API 주소 형식이 아닙니다.")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["offset"] = str(offset)
    query["limit"] = str(limit)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _review_key(review: dict, fallback: str) -> str:
    for field in ("id", "reviewId", "review_id", "uuid"):
        value = review.get(field)
        if value not in (None, ""):
            return f"{field}:{value}"
    return fallback


def _fetch_json(url: str, headers: dict[str, str], timeout: float) -> tuple[dict, int]:
    started = time.monotonic()
    req = Request(url, headers=headers, method="GET")
    with _open_no_redirect(req, timeout=timeout) as response:
        status = int(getattr(response, "status", 200))
        content_type = response.headers.get_content_type()
        if status != 200:
            raise RuntimeError(f"배민 API HTTP {status}")
        if content_type not in {"application/json", "text/json"}:
            raise RuntimeError(f"배민 API가 JSON이 아닌 응답을 반환했습니다: {content_type}")
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("배민 API 응답 형식이 올바르지 않습니다.")
    return payload, round((time.monotonic() - started) * 1000)


@revu_bp.get("/health")
def revu_health():
    key = _collector_key()
    return jsonify(
        ok=True,
        service="LUPER REVU collector",
        mode="stateless",
        authConfigured=bool(key and key != "change-me"),
        platforms={"baemin": "ready", "coupang": "next", "yogiyo": "next", "naver": "next"},
    )


@revu_bp.get("/auth-check")
def revu_auth_check():
    if not _authorized():
        return jsonify(ok=False, error="unauthorized"), 401
    return jsonify(ok=True, authenticated=True, service="LUPER REVU collector")


@revu_bp.post("/collect/baemin")
def collect_baemin():
    if not _authorized():
        return jsonify(ok=False, error="unauthorized"), 401

    body = request.get_json(silent=True) or {}
    raw_url = str(body.get("url", "")).strip()
    headers = _safe_headers(body.get("headers"))
    try:
        page_size = min(max(int(body.get("pageSize", 10)), 1), 50)
        max_pages = min(max(int(body.get("maxPages", 100)), 1), 200)
        timeout = min(max(float(body.get("timeoutSeconds", 25)), 3), 45)
        _baemin_url(raw_url, 0, page_size)
    except (TypeError, ValueError) as exc:
        return jsonify(ok=False, error="invalid_request", message=str(exc)), 400

    reviews = {}
    timings = []
    started = time.monotonic()
    offset = 0

    try:
        for page in range(1, max_pages + 1):
            page_url = _baemin_url(raw_url, offset, page_size)
            payload, response_ms = _fetch_json(page_url, headers, timeout)
            batch = payload.get("reviews") or []
            if not isinstance(batch, list):
                raise RuntimeError("배민 API reviews 값이 배열이 아닙니다.")
            for index, review in enumerate(batch):
                if isinstance(review, dict):
                    reviews[_review_key(review, f"{page}:{index}:{json.dumps(review, sort_keys=True, ensure_ascii=False)}")] = review
            timings.append({"page": page, "offset": offset, "received": len(batch), "responseMs": response_ms})
            has_next = bool(payload.get("next"))
            if not batch or not has_next:
                break
            offset += len(batch)
        else:
            return jsonify(
                ok=False,
                error="page_limit_reached",
                count=len(reviews),
                reviews=list(reviews.values()),
                timings=timings,
            ), 409
    except HTTPError as exc:
        return jsonify(ok=False, error="upstream_http_error", status=exc.code, message="배민 인증 또는 요청 권한을 확인해 주세요."), 502
    except (TimeoutError, URLError) as exc:
        return jsonify(ok=False, error="upstream_connection_error", message=str(getattr(exc, "reason", exc))), 504
    except (json.JSONDecodeError, RuntimeError) as exc:
        return jsonify(ok=False, error="upstream_response_error", message=str(exc)), 502

    return jsonify(
        ok=True,
        platform="baemin",
        count=len(reviews),
        pages=len(timings),
        elapsedMs=round((time.monotonic() - started) * 1000),
        timings=timings,
        reviews=list(reviews.values()),
        persisted=False,
    )
