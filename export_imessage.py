"""
Export macOS Messages from chat.db into:

- ``cpt_out.txt`` — one substantive message per line (CPT corpus).
- ``sft_output.json`` — JSON array of ``[system, user, assistant]`` turns for texting-style SFT.

Filters out group threads (many handles), tapback/reaction lines, and very short / one-word
messages so CPT/SFT match “texting yourself,” not noisy threads.

Requires Full Disk Access for the terminal running the export.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# Shared with ``chat.py`` / ``sft.py`` — same system string at train and inference.
TEXTING_SYSTEM_PROMPT = (
    "You are replying as yourself in a personal text thread. Write short, natural SMS-style "
    "messages: how you actually text—casual tone, no assistant disclaimers, no bullet lists "
    "unless you would really send that."
)

# Drop low-signal lines (noise) for CPT and SFT.
MIN_MESSAGE_WORDS = 2
MIN_MESSAGE_CHARS = 8

# Group / multi-party: chats linked to this many distinct handles are skipped entirely.
MIN_DISTINCT_HANDLES_FOR_GROUP = 3

# Tapback / reaction summary text sometimes stored as normal message rows.
_TAPBACK_PREFIXES = (
    "liked ",
    "loved ",
    "disliked ",
    "emphasized ",
    "questioned ",
    "laughed at ",
)

CPT_OUTPUT_FILE = Path("cpt_out.txt")
SFT_OUTPUT_FILE = Path("sft_output.json")
# Backward compatibility for older imports
OUTPUT_FILE = CPT_OUTPUT_FILE

PROGRESS_EVERY = 10_000


def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _db_uri(path: Path) -> str:
    parts = urlsplit(path.expanduser().resolve().as_uri())
    query = "mode=ro"
    if parts.query:
        query = parts.query + "&" + query
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return row is not None


def _group_chat_ids(conn: sqlite3.Connection) -> set[int]:
    """Chats with several participants (typical group / busy thread) — excluded from export."""
    out: set[int] = set()
    if not _has_table(conn, "chat_handle_join"):
        return out
    try:
        rows = conn.execute(
            f"""
            SELECT chat_id
            FROM chat_handle_join
            GROUP BY chat_id
            HAVING COUNT(DISTINCT handle_id) >= {MIN_DISTINCT_HANDLES_FOR_GROUP}
            """
        ).fetchall()
    except sqlite3.Error:
        return out
    for r in rows:
        try:
            out.add(int(r[0]))
        except (TypeError, ValueError):
            continue
    return out


def _reaction_filter_sql(conn: sqlite3.Connection) -> str:
    """Exclude tapback / reaction / sticker side-effect rows when the column exists."""
    if not _has_column(conn, "message", "associated_message_type"):
        return ""
    return (
        " AND (m.associated_message_type IS NULL "
        " OR m.associated_message_type < 2000 "
        " OR m.associated_message_type > 3006) "
    )


def _is_tapback_or_reaction_summary(text: str) -> bool:
    t = text.strip().lower()
    return any(t.startswith(p) for p in _TAPBACK_PREFIXES)


def _is_substantive_message(text: str) -> bool:
    """Drop one-word / very short replies and tapback summaries."""
    if _is_tapback_or_reaction_summary(text):
        return False
    words = [w for w in text.split() if w.strip()]
    if len(words) < MIN_MESSAGE_WORDS:
        return False
    if len(text.strip()) < MIN_MESSAGE_CHARS:
        return False
    # Standalone “k” / “ok” style after split edge cases
    if len(words) == MIN_MESSAGE_WORDS and all(len(w) <= 3 for w in words):
        return False
    return True


# Effect edits / invisible-only payloads (best-effort).
_EFFECT_RE = re.compile(
    r"^\s*(?:Edited to|Edited from|This message responded to with)\b",
    re.IGNORECASE,
)


def _is_noise_body(text: str) -> bool:
    if _EFFECT_RE.match(text):
        return True
    # Mostly punctuation / emoji-only one “token”
    stripped = text.strip()
    if not re.search(r"[A-Za-z0-9]", stripped) and len(stripped) <= 20:
        return True
    return False


def _compact_body(body: str) -> str | None:
    kept: list[str] = []
    for raw in body.splitlines():
        if not raw.strip():
            continue
        kept.append(raw.rstrip())
    if not kept:
        return None
    text = "\n".join(kept).strip()
    return text or None


def export_imessage() -> None:
    log("[export] iMessage → cpt_out.txt + sft_output.json (filtered: 1:1-style + substantive only)")
    log("[export] " + "-" * 40)

    db_path = Path.home() / "Library/Messages/chat.db"
    if not db_path.is_file():
        warn(f"Database not found: {db_path}")
        sys.exit(1)

    log(f"[export] Source DB: {db_path}")
    log(f"[export] CPT file:  {CPT_OUTPUT_FILE.resolve()}")
    log(f"[export] SFT file:  {SFT_OUTPUT_FILE.resolve()}")

    try:
        log("[export] Opening database (read-only)…")
        conn = sqlite3.connect(_db_uri(db_path), uri=True)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        warn(f"Could not open Messages database (permission denied?).\n{exc}")
        warn("Grant Full Disk Access to this terminal: System Settings → Privacy & Security.")
        sys.exit(1)

    reaction_where = _reaction_filter_sql(conn)
    skip_chat_ids = _group_chat_ids(conn)
    if skip_chat_ids:
        log(f"[export] Skipping {len(skip_chat_ids):,} multi-party / group chat id(s) (handle join).")

    log("[export] Counting chats and messages…")
    count_sql = f"""
        SELECT
            COUNT(DISTINCT c.ROWID) AS n_chats,
            COUNT(*) AS n_messages
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        JOIN chat c ON c.ROWID = cmj.chat_id
        WHERE 1=1 {reaction_where}
    """
    row = conn.execute(count_sql).fetchone()
    n_chats = int(row["n_chats"] or 0)
    n_messages = int(row["n_messages"] or 0)
    log(f"[export] Found {n_chats} conversation(s), {n_messages} message row(s).")
    if n_messages == 0:
        warn("No messages matched the export filter. Check Full Disk Access if this is unexpected.")
    log(
        "[export] Streaming rows (CPT lines + SFT examples; progress every "
        f"{PROGRESS_EVERY:,} rows + each new chat)…"
    )

    export_sql = f"""
        SELECT
            c.ROWID AS chat_id,
            IFNULL(c.display_name, c.chat_identifier) AS chat_title,
            c.chat_identifier,
            IFNULL(m.text, '') AS body,
            m.is_from_me AS is_from_me
        FROM message m
        JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
        JOIN chat c ON c.ROWID = cmj.chat_id
        WHERE 1=1 {reaction_where}
        ORDER BY c.ROWID ASC, m.date ASC, m.ROWID ASC
    """

    CPT_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    prev_chat_id: int | None = None
    chat_index = 0
    total_cpt = 0
    total_sft = 0
    skipped_group = 0
    skipped_noise = 0
    last_scanned = 0
    # Last substantive message body per chat (for SFT “incoming line → your reply”).
    prev_by_chat: dict[int, str] = {}

    sft_examples: list[dict[str, Any]] = []

    with CPT_OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as out_cpt:
        try:
            cursor = conn.execute(export_sql)
        except sqlite3.Error as exc:
            warn(f"Query failed: {exc}")
            sys.exit(1)

        for scanned, r in enumerate(cursor, start=1):
            cid = int(r["chat_id"])
            if cid != prev_chat_id:
                chat_index += 1
                prev_chat_id = cid
                title = (r["chat_title"] or r["chat_identifier"] or "?")[:72]
                log(f"[export]   Chat {chat_index}/{n_chats} (id={cid}) — {title}")

            if cid in skip_chat_ids:
                skipped_group += 1
                continue

            is_me = bool(r["is_from_me"])

            body = r["body"] or ""
            text = _compact_body(body)
            if text is None:
                continue
            if _is_noise_body(text) or not _is_substantive_message(text):
                skipped_noise += 1
                continue

            out_cpt.write(text + "\n")
            total_cpt += 1

            if is_me and cid in prev_by_chat:
                prev_text = prev_by_chat[cid]
                if _is_substantive_message(prev_text) and not _is_noise_body(prev_text):
                    # User turn = prior message body only (no handle/phone in the JSON).
                    user_content = prev_text
                    row_obj = {
                        "messages": [
                            {"role": "system", "content": TEXTING_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": text},
                        ]
                    }
                    sft_examples.append(row_obj)
                    total_sft += 1

            prev_by_chat[cid] = text

            last_scanned = scanned
            if scanned % PROGRESS_EVERY == 0 and n_messages:
                pct = 100.0 * scanned / n_messages
                log(
                    f"[export]   … scanned {scanned:,}/{n_messages:,} rows ({pct:.1f} %); "
                    f"cpt lines {total_cpt:,}, sft examples {total_sft:,}"
                )

    with SFT_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(sft_examples, f, ensure_ascii=False)

    conn.close()
    if last_scanned and (last_scanned < PROGRESS_EVERY or last_scanned % PROGRESS_EVERY != 0):
        log(
            f"[export]   Scan complete: {last_scanned:,} row(s); "
            f"cpt lines {total_cpt:,}, sft examples {total_sft:,}"
        )

    log("[export] Done.")
    log(f"[export]   CPT:  {CPT_OUTPUT_FILE.resolve()} ({total_cpt:,} lines)")
    log(f"[export]   SFT:  {SFT_OUTPUT_FILE.resolve()} ({total_sft:,} JSON examples)")
    if skipped_group or skipped_noise:
        log(
            f"[export]   Skipped rows: {skipped_group:,} (group thread) + "
            f"{skipped_noise:,} (short / tapback / noise)."
        )
    log(
        "[export]   SFT = [system, user, assistant] texting pairs in 1:1-style threads "
        "(see sft.py + chat.py)."
    )


run_export = export_imessage
