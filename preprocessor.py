import re
import pandas as pd
from datetime import datetime


# ──────────────────────────────────────────────
# CANONICAL FORMAT (what we split on directly)
# M/D/YY, HH:MM -   OR   M/D/YYYY, HH:MM -
# ──────────────────────────────────────────────
_CANONICAL = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s"

# ──────────────────────────────────────────────
# ALL SUPPORTED RAW FORMATS
# Each entry: (regex, strptime_formats_to_try)
# Order matters — more specific first
# ──────────────────────────────────────────────
_RAW_FORMATS = [
    # ── Already canonical (M/D/YY or M/D/YYYY, HH:MM - ) ──
    # Detected separately — no conversion needed

    # ── DD/MM/YYYY, HH:MM am/pm  (iOS India 4-digit year) ──
    (r"\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s-\s",
     ["%d/%m/%Y, %I:%M %p - "]),

    # ── DD/MM/YYYY, HH:MM  (Android India 4-digit year, 24h) ──
    (r"\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s-\s",
     ["%d/%m/%Y, %H:%M - "]),

    # ── DD/MM/YY, HH:MM am/pm  (iOS India 2-digit year) ──
    (r"\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s-\s",
     ["%d/%m/%y, %I:%M %p - "]),

    # ── M/D/YY, HH:MM am/pm  (iOS US) ──
    (r"\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s-\s",
     ["%m/%d/%y, %I:%M %p - "]),

    # ── DD.MM.YYYY, HH:MM  (German dot-separator) ──
    (r"\d{1,2}\.\d{1,2}\.\d{4},\s\d{1,2}:\d{2}\s-\s",
     ["%d.%m.%Y, %H:%M - "]),

    # ── YYYY-MM-DD, HH:MM  (ISO style) ──
    (r"\d{4}-\d{1,2}-\d{1,2},\s\d{1,2}:\d{2}\s-\s",
     ["%Y-%m-%d, %H:%M - "]),

    # ── DD/MM/YYYY, HH:MM:SS  (with seconds) ──
    (r"\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}:\d{2}\s-\s",
     ["%d/%m/%Y, %H:%M:%S - "]),
]


def _detect_format(data: str):
    """
    Returns (pattern, fmts) of the first format that matches 5+ times,
    or None if already canonical / unrecognised.
    """
    # Check if already canonical (M/D/YY, HH:MM - )
    canonical_hits = len(re.findall(_CANONICAL, data[:5000]))
    if canonical_hits >= 3:
        return None   # already fine, skip normalisation

    for pattern, fmts in _RAW_FORMATS:
        hits = re.findall(pattern, data[:5000], flags=re.IGNORECASE)
        if len(hits) >= 3:
            return pattern, fmts
    return None


def _normalise(data: str, pattern: str, fmts: list) -> str:
    """
    Convert matched timestamps to canonical M/D/YY, HH:MM - format.
    Only called when the file is NOT already canonical.
    """
    def replacer(match):
        raw = match.group(0)
        for fmt in fmts:
            try:
                dt = datetime.strptime(raw, fmt)
                # Use zero-padded month/day for consistent parsing later
                return dt.strftime("%m/%d/%y, %H:%M - ")
            except ValueError:
                continue
        return raw   # leave untouched if nothing works

    return re.sub(pattern, replacer, data, flags=re.IGNORECASE)


# ──────────────────────────────────────────────
# MAIN PREPROCESS
# ──────────────────────────────────────────────

def preprocess(data: str) -> pd.DataFrame:

    # Step 1 — normalise only if not already canonical
    fmt_info = _detect_format(data)
    if fmt_info is not None:
        pattern, fmts = fmt_info
        data = _normalise(data, pattern, fmts)

    # Step 2 — split on canonical pattern
    messages_raw = re.split(_CANONICAL, data)[1:]
    dates_raw    = re.findall(_CANONICAL, data)

    if not messages_raw:
        raise ValueError(
            "Could not parse any messages. "
            "Please make sure you exported the chat as plain text (without media)."
        )

    df = pd.DataFrame({"user_message": messages_raw, "message_date": dates_raw})

    # Step 3 — parse dates
    parsed = False
    for fmt in ("%m/%d/%y, %H:%M - ", "%m/%d/%Y, %H:%M - "):
        try:
            df["message_date"] = pd.to_datetime(df["message_date"], format=fmt)
            parsed = True
            break
        except Exception:
            continue
    if not parsed:
        try:
            df["message_date"] = pd.to_datetime(
                df["message_date"], infer_datetime_format=True
            )
        except Exception:
            raise ValueError(
                "Timestamps were found but could not be parsed. "
                "Please open an issue with a sample of your chat format."
            )

    df.rename(columns={"message_date": "date"}, inplace=True)

    # Step 4 — split user / message
    users, messages = [], []
    for msg in df["user_message"]:
        # Strip leading ' - ' that some formats leave behind
        msg = re.sub(r"^-\s", "", msg)
        parts = re.split(r"^([^:]+):\s", msg, maxsplit=1)
        if len(parts) >= 3:
            users.append(parts[1].strip())
            messages.append(parts[2])
        else:
            users.append("group_notification")
            messages.append(parts[0])

    df["user"]    = users
    df["message"] = messages
    df.drop(columns=["user_message"], inplace=True)

    # Step 5 — time feature columns
    df["year"]      = df["date"].dt.year
    df["only_date"] = df["date"].dt.date
    df["month_num"] = df["date"].dt.month
    df["month"]     = df["date"].dt.month_name()
    df["day"]       = df["date"].dt.day
    df["day_name"]  = df["date"].dt.day_name()
    df["hour"]      = df["date"].dt.hour
    df["minute"]    = df["date"].dt.minute

    def _period(h):
        nxt = "00" if h == 23 else str(h + 1)
        return f"{'00' if h == 0 else str(h)}-{nxt}"

    df["period"]         = df["hour"].apply(_period)
    df["message_length"] = df["message"].apply(lambda m: len(m.split()))
    df["is_media"]       = df["message"].str.strip().str.startswith("<Media omitted>")

    return df