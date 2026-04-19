"""
Export your public-channel messages via Slack search: ``from:me``, drop DMs / private channels.

Slack caps each search at 100 pages (~10k hits). We **shard by calendar month** and merge with dedupe.

**Default date range:** probes your **oldest** and **newest** ``from:me`` message (search sorted by
timestamp asc/desc, one hit each), then walks **every month** between those dates (UTC calendar days).

**Credentials** (set in ``slack/.env`` or the environment):

- ``SLACK_XOXC_TOKEN`` — Bearer token (``xoxc-…``) for ``Authorization``
- ``SLACK_D_COOKIE`` — Slack ``d`` cookie (``xoxd-…`` or URL-encoded value from devtools)

Optional overrides:

- ``SLACK_EXPORT_START`` / ``SLACK_EXPORT_END`` — ``YYYY-MM-DD`` (omit either side to probe it)
- ``SLACK_EXPORT_SHARD`` — ``month`` (default), ``week``, or ``day`` if a month hits the page cap

**Resume (does not wipe ``my_slack_messages.json``):**

- Existing non-empty JSON is **loaded** and deduped; new hits are appended.
- ``--start-from YYYY-MM-DD`` / ``SLACK_EXPORT_START_FROM`` — skip time shards that end on or before this date (e.g. after April, use ``2025-05-01`` to continue from May).
- ``--start-page N`` / ``SLACK_EXPORT_START_PAGE`` — for the **first** non-skipped shard only, begin at page ``N`` (e.g. ``39`` after a crash on page 39). Omit to refetch from page 1 (dedupe prevents duplicates).
- ``--fresh`` — ignore existing file and start an empty export (overwrites at end).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")

_DELAY = 1.2
_MAX_PAGE = 100  # Slack: max 100 pages × 100 results per search.messages

_AUTH_ERRORS = frozenset(
    {
        "invalid_auth",
        "not_authed",
        "token_revoked",
        "account_inactive",
        "token_expired",
    }
)


def _slack_credentials() -> tuple[dict[str, str], dict[str, str]]:
    token = (os.environ.get("SLACK_XOXC_TOKEN") or "").strip()
    cookie_d = (os.environ.get("SLACK_D_COOKIE") or "").strip()
    if not token or not cookie_d:
        raise SystemExit(
            "Set SLACK_XOXC_TOKEN and SLACK_D_COOKIE in slack/.env (or the environment). "
            "See exportSlackMessages.py module docstring."
        )
    headers = {"Authorization": f"Bearer {token}"}
    cookies = {"d": cookie_d}
    return headers, cookies


def _parse_date(s: str | None, default: date) -> date:
    if not s or not str(s).strip():
        return default
    parts = str(s).strip().split("-")
    if len(parts) != 3:
        raise ValueError(f"Expected YYYY-MM-DD, got {s!r}")
    y, m, d = (int(parts[0]), int(parts[1]), int(parts[2]))
    return date(y, m, d)


def _ts_to_date_utc(ts: str | float | None) -> date | None:
    if ts is None:
        return None
    try:
        sec = float(ts)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(sec, tz=timezone.utc).date()


def _search_edge_date(
    headers: dict[str, str],
    cookies: dict[str, str],
    *,
    oldest: bool,
) -> date | None:
    """Single hit: globally oldest or newest ``from:me`` message (for date-range bounds)."""

    data = _api(
        headers,
        cookies,
        "search.messages",
        {
            "query": "from:me",
            "count": "1",
            "page": "1",
            "sort": "timestamp",
            "sort_dir": "asc" if oldest else "desc",
        },
    )
    if not data.get("ok"):
        print(f"[export] edge probe ({'oldest' if oldest else 'newest'}): {data.get('error')}", flush=True)
        return None
    matches = (data.get("messages") or {}).get("matches") or []
    if not matches:
        return None
    return _ts_to_date_utc(matches[0].get("ts"))


def _resolve_export_bounds(headers: dict[str, str], cookies: dict[str, str]) -> tuple[date, date]:
    start_e = os.environ.get("SLACK_EXPORT_START")
    end_e = os.environ.get("SLACK_EXPORT_END")

    if start_e and end_e:
        d0 = _parse_date(start_e, date.today())
        d1 = _parse_date(end_e, date.today())
    elif start_e:
        d0 = _parse_date(start_e, date.today())
        print("[export] Probing newest `from:me` date for end of range…", flush=True)
        d1 = _search_edge_date(headers, cookies, oldest=False) or date.today()
    elif end_e:
        print("[export] Probing oldest `from:me` date for start of range…", flush=True)
        d0 = _search_edge_date(headers, cookies, oldest=True) or date(2015, 1, 1)
        d1 = _parse_date(end_e, date.today())
    else:
        print("[export] Probing oldest and newest `from:me` message dates (UTC)…", flush=True)
        d0 = _search_edge_date(headers, cookies, oldest=True) or date(2015, 1, 1)
        d1 = _search_edge_date(headers, cookies, oldest=False) or date.today()
        print(f"[export] Range from probe: {d0.isoformat()} … {d1.isoformat()}", flush=True)

    if d0 > d1:
        d0, d1 = d1, d0
    return d0, d1


def _next_month_first(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _iter_shards_month(d0: date, d1: date):
    """Yield (label, after inclusive, before exclusive) for each calendar slice."""

    cur_first = date(d0.year, d0.month, 1)
    while cur_first <= d1:
        nxt = _next_month_first(cur_first)
        lo = max(cur_first, d0)
        hi_excl = min(nxt, d1 + timedelta(days=1))
        if lo < hi_excl:
            label = f"{lo.isoformat()}…{(hi_excl - timedelta(days=1)).isoformat()}"
            yield label, lo, hi_excl
        cur_first = nxt


def _iter_shards_week(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        nxt = cur + timedelta(days=7)
        hi_excl = min(nxt, d1 + timedelta(days=1))
        if cur < hi_excl:
            label = f"{cur.isoformat()}…{(hi_excl - timedelta(days=1)).isoformat()}"
            yield label, cur, hi_excl
        cur = nxt


def _iter_shards_day(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        hi_excl = cur + timedelta(days=1)
        if cur < hi_excl:
            yield cur.isoformat(), cur, hi_excl
        cur = hi_excl


def _iter_shards(kind: str, d0: date, d1: date):
    k = kind.strip().lower()
    if k == "month":
        yield from _iter_shards_month(d0, d1)
    elif k == "week":
        yield from _iter_shards_week(d0, d1)
    elif k == "day":
        yield from _iter_shards_day(d0, d1)
    else:
        raise ValueError(f"SLACK_EXPORT_SHARD must be month, week, or day; got {kind!r}")


def _retry_after_seconds(response: requests.Response) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 1.0
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 1.0


def _prompt_new_credentials(headers: dict[str, str], cookies: dict[str, str]) -> None:
    print(
        "\n[export] Slack rejected this session. Paste a fresh `d` cookie from your browser "
        "(Application → Cookies → slack.com, or the long xoxd-… value).\n"
        "Optional second line: new xoxc token if cookie alone is not enough (or press Enter to skip).\n"
        "Empty line = exit.\n",
        flush=True,
    )
    cookie_line = input("d cookie> ").strip()
    if not cookie_line:
        raise SystemExit("[export] Aborted (no cookie).")
    cookies["d"] = cookie_line
    token_line = input("xoxc token (Enter to keep)> ").strip()
    if token_line:
        headers["Authorization"] = f"Bearer {token_line}"


def _api(
    headers: dict[str, str],
    cookies: dict[str, str],
    method: str,
    data: dict,
) -> dict:
    while True:
        time.sleep(_DELAY)
        r = requests.post(
            f"https://slack.com/api/{method}",
            headers=headers,
            cookies=cookies,
            data=data,
            timeout=60,
        )
        try:
            payload = r.json()
        except json.JSONDecodeError:
            print(f"[export] Non-JSON response (status {r.status_code}), retrying…", flush=True)
            time.sleep(1.0)
            continue

        if payload.get("ok"):
            return payload

        err = payload.get("error") or ""

        if err == "ratelimited" or r.status_code == 429:
            wait = _retry_after_seconds(r)
            print(f"[export] Rate limited — waiting {wait:.1f}s then retrying…", flush=True)
            time.sleep(wait)
            continue

        if err in _AUTH_ERRORS or r.status_code in (401, 403):
            print(f"[export] Auth error: {err or r.status_code}", flush=True)
            _prompt_new_credentials(headers, cookies)
            continue

        return payload


def _public_hit(m: dict) -> bool:
    ch = m.get("channel") or {}
    return not (ch.get("is_im") or ch.get("is_mpim") or ch.get("is_private"))


def _msg_key(m: dict) -> tuple[str, str]:
    return ((m.get("channel") or {}).get("id") or "", str(m.get("ts") or ""))


def _write_json_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def _parse_start_from(raw: str | None) -> date | None:
    if not raw or not str(raw).strip():
        return None
    return _parse_date(str(raw).strip(), date.today())


def _shard_skipped_by_start_from(hi_excl: date, start_from: date | None) -> bool:
    """Skip shards whose window ends on/before ``start_from`` (monthly: omit earlier months)."""

    if start_from is None:
        return False
    return hi_excl <= start_from


def _load_existing_export(path: Path) -> tuple[list[dict], set[tuple[str, str]]]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[export] Could not read {path.name}: {exc}; starting empty.", flush=True)
        return rows, seen
    if not isinstance(data, list):
        return rows, seen
    for m in data:
        if not isinstance(m, dict):
            continue
        if not _public_hit(m):
            continue
        k = _msg_key(m)
        if k in seen:
            continue
        seen.add(k)
        rows.append(m)
    return rows, seen


def export_slack_messages(
    *,
    start_from: date | str | None = None,
    start_page: int | None = None,
    load_existing: bool = True,
    fresh: bool = False,
) -> None:
    headers, cookies = _slack_credentials()

    d0, d1 = _resolve_export_bounds(headers, cookies)

    shard = os.environ.get("SLACK_EXPORT_SHARD", "month")

    # CLI kwargs override env when passed explicitly; env used when None from CLI path.
    sf = start_from
    if sf is None and os.environ.get("SLACK_EXPORT_START_FROM"):
        sf = os.environ.get("SLACK_EXPORT_START_FROM")
    if isinstance(sf, str):
        try:
            sf = _parse_start_from(sf)
        except ValueError as exc:
            raise SystemExit(f"[export] Invalid start_from (use YYYY-MM-DD): {exc}") from exc
    elif sf is not None and not isinstance(sf, date):
        sf = None

    sp = start_page
    if sp is None and os.environ.get("SLACK_EXPORT_START_PAGE"):
        try:
            sp = int(os.environ["SLACK_EXPORT_START_PAGE"].strip())
        except ValueError:
            sp = None
    if sp is not None and sp < 1:
        sp = 1

    out = _SCRIPT_DIR / "my_slack_messages.json"
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()

    if fresh:
        print(f"[export] --fresh: ignoring existing {out.name}", flush=True)
    elif load_existing and out.is_file():
        rows, seen = _load_existing_export(out)
        if rows:
            print(
                f"[export] Loaded {len(rows)} existing public message(s) from {out.name} (dedupe active; not resetting).",
                flush=True,
            )
    else:
        print(f"[export] No existing data loaded; new file → {out.name}", flush=True)

    print(
        f"[export] Date range {d0.isoformat()} … {d1.isoformat()} (shard={shard}); "
        f"each shard uses after:/before: + from:me",
        flush=True,
    )
    if sf is not None:
        print(
            f"[export] start_from={sf.isoformat()} — skipping shards with hi_excl <= this date",
            flush=True,
        )
    if sp is not None:
        print(
            f"[export] start_page={sp} — first non-skipped shard begins at this page",
            flush=True,
        )

    try:
        first_shard = True
        for label, lo, hi_excl in _iter_shards(shard, d0, d1):
            if _shard_skipped_by_start_from(hi_excl, sf):
                print(f"[export] [{label}] skipped (before start_from)", flush=True)
                continue

            query = f"from:me after:{lo.isoformat()} before:{hi_excl.isoformat()}"
            page = sp if first_shard and sp is not None else 1
            if first_shard:
                first_shard = False
            pages = 1

            while page <= _MAX_PAGE and page <= pages:
                data = _api(
                    headers,
                    cookies,
                    "search.messages",
                    {
                        "query": query,
                        "count": "100",
                        "page": str(page),
                        "sort": "timestamp",
                        "sort_dir": "asc",
                    },
                )
                if not data.get("ok"):
                    print(
                        f"[export] shard {label!r}: search.messages {data.get('error')}",
                        flush=True,
                    )
                    break

                block = data.get("messages") or {}
                matches = block.get("matches") or []
                try:
                    pages = int((block.get("paging") or {}).get("pages") or 1)
                except (TypeError, ValueError):
                    pages = 1

                for m in matches:
                    if not _public_hit(m):
                        continue
                    k = _msg_key(m)
                    if k in seen:
                        continue
                    seen.add(k)
                    rows.append(m)

                _write_json_atomic(out, rows)

                total = block.get("total")
                if page == 1 and total is not None:
                    print(
                        f"[export] [{label}] index reports {total} hit(s) in this window",
                        flush=True,
                    )

                print(
                    f"[export] [{label}] page {page}/{min(pages, _MAX_PAGE)} "
                    f"(index {pages} pgs) — {len(rows)} public total → {out.name}",
                    flush=True,
                )

                if pages > _MAX_PAGE:
                    print(
                        f"[export] WARNING: shard {label!r} still needs >{_MAX_PAGE} pages; "
                        f"set SLACK_EXPORT_SHARD=week or day and re-run (dedupe keeps one copy).",
                        flush=True,
                    )

                if not matches:
                    break
                page += 1

    except KeyboardInterrupt:
        print("\n[export] Stopped early; keeping partial file.", flush=True)
        raise
    finally:
        rows.sort(key=lambda m: float(m.get("ts") or 0))
        _write_json_atomic(out, rows)

    print(f"[export] done — {len(rows)} message(s) → {out}", flush=True)


def parse_export_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Slack search hits to my_slack_messages.json (month-sharded, resumable).",
    )
    p.add_argument(
        "--start-from",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Skip shards ending on/before this date (e.g. 2025-05-01 to continue from May). "
        "Env: SLACK_EXPORT_START_FROM",
    )
    p.add_argument(
        "--start-page",
        type=int,
        default=None,
        metavar="N",
        help="First page for the first non-skipped shard after disk/API errors (e.g. 39). "
        "Env: SLACK_EXPORT_START_PAGE",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Do not load existing JSON; export from scratch (still writes atomically at end).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_export_args()
    export_slack_messages(
        start_from=args.start_from,
        start_page=args.start_page,
        load_existing=not args.fresh,
        fresh=args.fresh,
    )
