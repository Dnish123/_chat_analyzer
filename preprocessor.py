import re
import pandas as pd
from datetime import datetime

# ── Canonical format: M/D/YY, HH:MM -  or  M/D/YYYY, HH:MM - ────────────────
_CANONICAL = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s"

# ── All supported raw formats (more specific first) ───────────────────────────
_RAW_FORMATS = [
    # iOS India 4-digit year with am/pm
    (r"\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s-\s",
     ["%d/%m/%Y, %I:%M %p - "]),
    # Android India 4-digit year 24h
    (r"\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s-\s",
     ["%d/%m/%Y, %H:%M - "]),
    # iOS India 2-digit year with am/pm
    (r"\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s-\s",
     ["%d/%m/%y, %I:%M %p - "]),
    # iOS US 2-digit year with am/pm
    (r"\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s(?:am|pm|AM|PM)\s-\s",
     ["%m/%d/%y, %I:%M %p - "]),
    # German dot-separator
    (r"\d{1,2}\.\d{1,2}\.\d{4},\s\d{1,2}:\d{2}\s-\s",
     ["%d.%m.%Y, %H:%M - "]),
    # ISO style
    (r"\d{4}-\d{1,2}-\d{1,2},\s\d{1,2}:\d{2}\s-\s",
     ["%Y-%m-%d, %H:%M - "]),
    # With seconds
    (r"\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}:\d{2}\s-\s",
     ["%d/%m/%Y, %H:%M:%S - "]),
]


def _detect_format(data: str):
    """Return (pattern, fmts) if conversion needed, None if already canonical."""
    if len(re.findall(_CANONICAL, data[:5000])) >= 3:
        return None
    for pattern, fmts in _RAW_FORMATS:
        if len(re.findall(pattern, data[:5000], flags=re.IGNORECASE)) >= 3:
            return pattern, fmts
    return None


def _normalise(data: str, pattern: str, fmts: list) -> str:
    def replacer(match):
        raw = match.group(0)
        for fmt in fmts:
            try:
                return datetime.strptime(raw, fmt).strftime("%m/%d/%y, %H:%M - ")
            except ValueError:
                continue
        return raw
    return re.sub(pattern, replacer, data, flags=re.IGNORECASE)


def preprocess(data: str) -> pd.DataFrame:
    # Step 1 — normalise only if needed
    fmt_info = _detect_format(data)
    if fmt_info is not None:
        data = _normalise(data, *fmt_info)

    # Step 2 — split on canonical pattern
    messages_raw = re.split(_CANONICAL, data)[1:]
    dates_raw    = re.findall(_CANONICAL, data)

    if not messages_raw:
        raise ValueError(
            "Could not parse any messages. "
            "Please export the chat as plain text (without media) from WhatsApp."
        )

    df = pd.DataFrame({"user_message": messages_raw, "message_date": dates_raw})

    # Step 3 — parse dates (no deprecated infer_datetime_format)
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
            # pandas ≥2.0 compatible fallback
            df["message_date"] = pd.to_datetime(df["message_date"], format="mixed")
        except Exception:
            raise ValueError(
                "Timestamps found but could not be parsed. "
                "Please share a sample of your chat format."
            )

    df.rename(columns={"message_date": "date"}, inplace=True)

    # Step 4 — split user / message
    users, messages = [], []
    for msg in df["user_message"]:
        msg   = re.sub(r"^-\s", "", msg)          # strip leading " - " artifact
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