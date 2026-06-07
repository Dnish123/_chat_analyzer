"""
helper.py — analytics functions for WhatsApp Chat Analyzer
"""

import re
import os
import json
import hashlib
import datetime
from collections import Counter
from pathlib import Path

import emoji
import pandas as pd
from wordcloud import WordCloud

# ── Optional cloud storage (Supabase free tier) ───────────────────────────────
# Set env vars SUPABASE_URL and SUPABASE_KEY to enable.
# Falls back silently to local disk if not configured.
try:
    from supabase import create_client
    _SUPA_URL = os.getenv("SUPABASE_URL", "")
    _SUPA_KEY = os.getenv("SUPABASE_ANON_KEY", "")
    _supabase = create_client(_SUPA_URL, _SUPA_KEY) if (_SUPA_URL and _SUPA_KEY) else None
except ImportError:
    _supabase = None

_LOCAL_STORE = Path("uploaded_chats")


def _file_id(raw_bytes: bytes) -> str:
    """SHA-256 fingerprint of the raw chat bytes."""
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def save_chat(raw_bytes: bytes, filename: str) -> dict:
    """
    Persist the raw chat file.
    1. Try Supabase Storage (free tier, up to 1 GB).
    2. Fall back to local disk under ./uploaded_chats/.
    Returns a dict with keys: storage, path, file_id.
    """
    fid  = _file_id(raw_bytes)
    dest = f"{fid}_{filename}"

    # ── Supabase ──
    if _supabase:
        try:
            bucket = "whatsapp-chats"
            _supabase.storage.from_(bucket).upload(
                dest, raw_bytes,
                file_options={"content-type": "text/plain"}
            )
            return {"storage": "supabase", "path": dest, "file_id": fid}
        except Exception:
            pass   # fall through to local

    # ── Local disk ──
    _LOCAL_STORE.mkdir(exist_ok=True)
    (LOCAL_STORE / dest).write_bytes(raw_bytes)
    return {"storage": "local", "path": str(_LOCAL_STORE / dest), "file_id": fid}


# ── Stop-words ────────────────────────────────────────────────────────────────

_STOP_WORDS: set[str] = set()

def _load_stop_words() -> set[str]:
    global _STOP_WORDS
    if _STOP_WORDS:
        return _STOP_WORDS
    for candidate in ["stop_hinglish.txt", "stop_words.txt"]:
        p = Path(candidate)
        if p.exists():
            _STOP_WORDS = set(p.read_text(encoding="utf-8").split())
            return _STOP_WORDS
    # Bare-minimum English fallback so the app never crashes
    _STOP_WORDS = {
        "the","a","an","is","it","in","on","at","to","of","and","or",
        "for","with","this","that","was","are","be","have","has","had",
        "i","me","my","we","our","you","your","he","she","they","them",
        "his","her","their","will","would","could","should","do","did",
        "not","no","yes","ok","okay","hi","hello","hey","lol","haha",
        "just","like","so","but","if","from","by","about","its","its",
        "what","when","where","who","how","why","there","here","time",
        "media","omitted","deleted","edited","message","pm","am",
    }
    return _STOP_WORDS


# ── Core statistics ───────────────────────────────────────────────────────────

def fetch_stats(selected_user: str, df: pd.DataFrame) -> dict:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    num_messages  = df.shape[0]
    num_words     = df["message"].apply(lambda m: len(m.split())).sum()
    num_media     = df["message"].str.contains("Media omitted", case=False, na=False).sum()
    num_links     = df["message"].str.contains(
                        r"https?://", regex=True, case=False, na=False
                    ).sum()
    avg_msg_len   = round(df["message"].apply(lambda m: len(m.split())).mean(), 1)

    all_emojis = [
        ch for msg in df["message"] for ch in msg if ch in emoji.EMOJI_DATA
    ]
    total_emojis = len(all_emojis)
    top_emoji    = Counter(all_emojis).most_common(1)[0][0] if all_emojis else "—"

    return {
        "num_messages":  num_messages,
        "num_words":     int(num_words),
        "num_media":     int(num_media),
        "num_links":     int(num_links),
        "avg_msg_len":   avg_msg_len,
        "total_emojis":  total_emojis,
        "top_emoji":     top_emoji,
    }


# ── User activity ─────────────────────────────────────────────────────────────

def most_busy_users(df: pd.DataFrame):
    counts = df["user"].value_counts().head(10)
    pct_df = (
        df["user"].value_counts(normalize=True)
        .mul(100).round(2)
        .reset_index()
        .rename(columns={"user": "name", "proportion": "percent",
                         "count": "percent"})   # pandas ≥2.0 uses "proportion"
    )
    # Normalise column names regardless of pandas version
    pct_df.columns = ["name", "percent"]
    return counts, pct_df


# ── Word cloud ────────────────────────────────────────────────────────────────

def create_wordcloud(selected_user: str, df: pd.DataFrame):
    stop_words = _load_stop_words()
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["user"] != "group_notification"].copy()
    temp = temp[~temp["message"].str.contains(
        "Media omitted|deleted|edited", case=False, na=False
    )]

    def _clean(msg: str) -> str:
        msg = re.sub(r"@\w+|<[^>]+>|https?://\S+", " ", msg)
        msg = re.sub(r"[^a-zA-Z\s]", " ", msg)
        return " ".join(w for w in msg.lower().split() if w not in stop_words)

    text = temp["message"].apply(_clean).str.cat(sep=" ").strip()
    if not text:
        text = "no words found"

    wc = WordCloud(
        width=800, height=400, min_font_size=8,
        background_color="white", colormap="viridis",
        max_words=150, collocations=False,
    )
    return wc.generate(text)


# ── Most common words ─────────────────────────────────────────────────────────

def most_common_words(selected_user: str, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    stop_words = _load_stop_words()
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    temp = df[df["user"] != "group_notification"].copy()
    temp = temp[~temp["message"].str.contains(
        "Media omitted|deleted|edited", case=False, na=False
    )]
    usernames = {u.lower() for u in df["user"].unique()}

    words = []
    for msg in temp["message"]:
        msg = re.sub(r"@\w+|<[^>]+>|https?://\S+", " ", msg)
        msg = re.sub(r"[^a-zA-Z\s]", " ", msg)
        for w in msg.lower().split():
            if w not in stop_words and w not in usernames and len(w) > 2:
                words.append(w)

    top = Counter(words).most_common(n)
    if not top:
        return pd.DataFrame(columns=["word", "count"])
    return pd.DataFrame(top, columns=["word", "count"])


# ── Emoji analysis ────────────────────────────────────────────────────────────

def emoji_analysis(selected_user: str, df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    all_emojis = [ch for msg in df["message"] for ch in msg if ch in emoji.EMOJI_DATA]
    if not all_emojis:
        return pd.DataFrame(columns=["emoji", "count"])
    top = Counter(all_emojis).most_common(n)
    return pd.DataFrame(top, columns=["emoji", "count"])


# ── Timelines ─────────────────────────────────────────────────────────────────

def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    tl = (
        df.groupby(["year", "month_num", "month"])
        .size()
        .reset_index(name="message")
    )
    tl["time"] = tl["month"] + "-" + tl["year"].astype(str)
    return tl


def daily_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    return df.groupby("only_date").size().reset_index(name="message")


# ── Activity maps ─────────────────────────────────────────────────────────────

_DAY_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
_MONTH_ORDER = ["January","February","March","April","May","June",
                "July","August","September","October","November","December"]

def week_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    counts = df["day_name"].value_counts()
    return counts.reindex(_DAY_ORDER, fill_value=0)


def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    counts = df["month"].value_counts()
    return counts.reindex(_MONTH_ORDER, fill_value=0).dropna()


def activity_heatmap(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    hm = df.pivot_table(
        index="day_name", columns="period",
        values="message", aggfunc="count"
    ).fillna(0)
    return hm.reindex([d for d in _DAY_ORDER if d in hm.index])


# ── Response time analysis ────────────────────────────────────────────────────

def response_time_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average response time (minutes) per user.
    Excludes group_notification and gaps > 24 h (new conversations).
    """
    d = df[df["user"] != "group_notification"].copy().sort_values("date")
    d["prev_user"] = d["user"].shift(1)
    d["prev_time"] = d["date"].shift(1)
    d["gap_min"]   = (d["date"] - d["prev_time"]).dt.total_seconds() / 60

    # Keep only genuine replies (different user, gap < 24 h)
    replies = d[(d["user"] != d["prev_user"]) & (d["gap_min"] < 1440)]
    if replies.empty:
        return pd.DataFrame(columns=["user", "avg_response_min"])
    return (
        replies.groupby("user")["gap_min"]
        .mean().round(1)
        .reset_index()
        .rename(columns={"gap_min": "avg_response_min"})
        .sort_values("avg_response_min")
    )


# ── Sentiment (lightweight, no ML deps) ──────────────────────────────────────

_POS = {
    "good","great","love","happy","awesome","amazing","nice","wonderful",
    "best","excellent","fantastic","perfect","thanks","thank","congrats",
    "congratulations","yes","haha","lol","hehe","😂","❤","😍","🎉","👍",
    "brilliant","superb","beautiful","enjoy","enjoyed","fun","yay",
}
_NEG = {
    "bad","sad","hate","angry","terrible","awful","worst","horrible",
    "no","nope","disappoint","sorry","sick","tired","miss","😢","😭",
    "😡","💔","annoying","annoyed","upset","problem","issue","fail",
    "failed","frustrate","frustrated","boring","bored","unfortunately",
}

# ── AI Chat Summary & Relationship Analysis ──────────────────────────────────

def _sample_messages(df: pd.DataFrame, max_messages: int = 300) -> list[dict]:
    """
    Returns a representative sample of messages as list of {user, message, date}.
    Takes evenly-spaced rows so we cover the full timeline, not just the start.
    """
    real = df[
        (df["user"] != "group_notification") &
        (~df["message"].str.contains("Media omitted|deleted|edited", case=False, na=False))
    ].copy()

    if len(real) > max_messages:
        indices = list(range(0, len(real), max(1, len(real) // max_messages)))[:max_messages]
        real = real.iloc[indices]

    return [
        {
            "user": row["user"],
            "message": row["message"].strip(),
            "date": str(row["only_date"]),
        }
        for _, row in real.iterrows()
    ]


def build_chat_context(df: pd.DataFrame) -> str:
    """
    Build a compact text block of sampled messages to feed to the AI.
    Format: [DATE] User: message
    """
    samples = _sample_messages(df)
    lines = [f"[{s['date']}] {s['user']}: {s['message']}" for s in samples]
    return "\n".join(lines)


def compute_relationship_signals(df: pd.DataFrame) -> dict:
    """
    Pure-Python signals (no AI) that enrich the AI prompt with hard numbers.
    Returns a dict of per-pair and per-user stats.
    """
    real = df[
        (df["user"] != "group_notification") &
        (~df["message"].str.contains("Media omitted|deleted|edited", case=False, na=False))
    ].copy().sort_values("date")

    users = [u for u in real["user"].unique()]
    total = len(real)

    # ── Per-user share ──
    share = (real["user"].value_counts() / total * 100).round(1).to_dict()

    # ── Reply patterns (who replies to whom) ──
    real["prev_user"] = real["user"].shift(1)
    real["gap_min"]   = (real["date"] - real["date"].shift(1)).dt.total_seconds() / 60
    replies = real[
        (real["user"] != real["prev_user"]) &
        (real["gap_min"] < 60)                # within same conversation window
    ]
    reply_counts: dict[tuple, int] = {}
    for _, row in replies.iterrows():
        pair = (row["prev_user"], row["user"])
        reply_counts[pair] = reply_counts.get(pair, 0) + 1

    # ── Pet names / terms of endearment ──
    ENDEARMENTS = {
        "babe","baby","love","dear","sweetheart","honey","darling","cutie",
        "jaan","yaar","bro","bhai","dude","sis","bestie","buddy","bff",
        "sir","boss","chief","team","guys","everyone","all",
    }
    endear_hits: dict[str, Counter] = {u: Counter() for u in users}
    for _, row in real.iterrows():
        tokens = set(re.sub(r"[^a-z\s]", " ", row["message"].lower()).split())
        hits = tokens & ENDEARMENTS
        for h in hits:
            endear_hits[row["user"]][h] += 1

    # ── Question asking (who asks the most questions?) ──
    real["is_question"] = real["message"].str.contains(r"\?", na=False)
    q_share = (real.groupby("user")["is_question"].sum() / real["is_question"].sum() * 100).round(1).to_dict()

    # ── Late-night messages (10pm–4am) ──
    late = real[real["hour"].between(22, 23) | real["hour"].between(0, 4)]
    late_share = (late["user"].value_counts() / len(late) * 100).round(1).to_dict() if len(late) else {}

    # ── Avg message length per user ──
    avg_len = real.groupby("user")["message"].apply(lambda x: x.str.split().apply(len).mean()).round(1).to_dict()

    return {
        "users": users,
        "message_share": share,
        "reply_pairs": {f"{a}→{b}": c for (a, b), c in sorted(reply_counts.items(), key=lambda x: -x[1])},
        "endearments_per_user": {u: dict(c.most_common(5)) for u, c in endear_hits.items()},
        "question_share": q_share,
        "late_night_share": late_share,
        "avg_message_length": avg_len,
        "date_range": f"{df['only_date'].min()} to {df['only_date'].max()}",
        "total_messages": total,
        "num_participants": len(users),
    }


def sentiment_over_time(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    temp = df[df["user"] != "group_notification"].copy()

    def _score(msg: str) -> float:
        tokens = set(re.sub(r"[^a-zA-Z\s]", " ", msg).lower().split()) | set(msg)
        pos = len(tokens & _POS)
        neg = len(tokens & _NEG)
        return pos - neg

    temp["sentiment"] = temp["message"].apply(_score)
    monthly = (
        temp.groupby(["year", "month_num", "month"])["sentiment"]
        .mean().round(2).reset_index()
    )
    monthly["time"] = monthly["month"] + "-" + monthly["year"].astype(str)
    return monthly