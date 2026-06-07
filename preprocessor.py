import re
import pandas as pd
from datetime import datetime


# ──────────────────────────────────────────────
# FORMAT DETECTION & NORMALISATION
# ──────────────────────────────────────────────

# Covers every WhatsApp timestamp variant seen in the wild:
#   • DD/MM/YYYY, HH:MM  (24h, no am/pm)  — Android India / EU
#   • DD/MM/YYYY, hh:mm am/pm             — iOS India
#   • M/D/YY, HH:MM                       — Android US
#   • M/D/YY, h:mm AM/PM                  — iOS US
#   • DD.MM.YYYY, HH:MM                   — German / dot-separator
#   • YYYY-MM-DD, HH:MM                   — ISO-style
#   • DD/MM/YY, HH:MM                     — 2-digit year

_TS_PATTERNS = [
    # Android/iOS India (DD/MM/YYYY)
    (r"(\d{1,2}/\d{1,2}/\d{4}),?\s(\d{1,2}:\d{2})\s?(am|pm|AM|PM)?",
     ["%d/%m/%Y, %I:%M %p", "%d/%m/%Y, %H:%M"]),
    # Android/iOS US (M/D/YY)
    (r"(\d{1,2}/\d{1,2}/\d{2}),?\s(\d{1,2}:\d{2})\s?(am|pm|AM|PM)?",
     ["%m/%d/%y, %I:%M %p", "%m/%d/%y, %H:%M"]),
    # German dot-separator (DD.MM.YYYY)
    (r"(\d{1,2}\.\d{1,2}\.\d{4}),?\s(\d{1,2}:\d{2})\s?(am|pm|AM|PM)?",
     ["%d.%m.%Y, %I:%M %p", "%d.%m.%Y, %H:%M"]),
    # ISO (YYYY-MM-DD)
    (r"(\d{4}-\d{1,2}-\d{1,2}),?\s(\d{1,2}:\d{2})\s?(am|pm|AM|PM)?",
     ["%Y-%m-%d, %I:%M %p", "%Y-%m-%d, %H:%M"]),
]

# Canonical split-pattern — after normalisation every message starts with this
_CANONICAL = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s"


def _try_parse(date_str: str, time_str: str, ampm: str | None, fmts: list[str]) -> datetime | None:
    combined = f"{date_str}, {time_str}"
    if ampm:
        combined += f" {ampm.upper()}"
    for fmt in fmts:
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def _normalise_timestamps(data: str) -> str:
    """Convert any supported timestamp format to M/D/YY, HH:MM - """
    for pattern, fmts in _TS_PATTERNS:
        def replacer(match, fmts=fmts):
            date_str = match.group(1)
            time_str = match.group(2)
            ampm = match.group(3) if match.lastindex >= 3 else None
            dt = _try_parse(date_str, time_str, ampm, fmts)
            if dt is None:
                return match.group(0)          # leave untouched if we can't parse
            return dt.strftime("%-m/%-d/%y, %H:%M - ")
        data = re.sub(pattern, replacer, data, flags=re.IGNORECASE)
    return data


# ──────────────────────────────────────────────
# MAIN PREPROCESS
# ──────────────────────────────────────────────

def preprocess(data: str) -> pd.DataFrame:
    data = _normalise_timestamps(data)

    messages_raw = re.split(_CANONICAL, data)[1:]
    dates_raw    = re.findall(_CANONICAL, data)

    if not messages_raw:
        raise ValueError(
            "Could not parse any messages. "
            "Please make sure you exported the chat without media (plain .txt)."
        )

    df = pd.DataFrame({"user_message": messages_raw, "message_date": dates_raw})

    # Parse dates — try two format strings to handle 1- vs 2-digit month/day
    for fmt in ("%m/%d/%y, %H:%M - ", "%-m/%-d/%y, %H:%M - "):
        try:
            df["message_date"] = pd.to_datetime(df["message_date"], format=fmt)
            break
        except Exception:
            continue
    else:
        df["message_date"] = pd.to_datetime(df["message_date"], infer_datetime_format=True)

    df.rename(columns={"message_date": "date"}, inplace=True)

    # ── Split user / message ──
    users, messages = [], []
    for msg in df["user_message"]:
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

    # ── Time features ──
    df["year"]      = df["date"].dt.year
    df["only_date"] = df["date"].dt.date
    df["month_num"] = df["date"].dt.month
    df["month"]     = df["date"].dt.month_name()
    df["day"]       = df["date"].dt.day
    df["day_name"]  = df["date"].dt.day_name()
    df["hour"]      = df["date"].dt.hour
    df["minute"]    = df["date"].dt.minute

    # ── Hour period labels ──
    def _period(h):
        nxt = "00" if h == 23 else str(h + 1)
        return f"{'00' if h == 0 else str(h)}-{nxt}"

    df["period"] = df["hour"].apply(_period)

    # ── Extra derived columns ──
    df["message_length"] = df["message"].apply(lambda m: len(m.split()))
    df["is_media"]       = df["message"].str.strip().eq("<Media omitted>") | \
                           df["message"].str.strip().eq("<Media omitted>\n")

    return df