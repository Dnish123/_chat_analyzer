"""
helper.py — analytics functions for WhatsApp Chat Analyzer
"""

import re
import os
import hashlib
from collections import Counter
from pathlib import Path

import emoji
import pandas as pd
from wordcloud import WordCloud

# ── Optional Supabase storage ─────────────────────────────────────────────────
# Keys are read lazily (inside save_chat) so that load_dotenv() in main.py
# always runs BEFORE we touch os.getenv — fixes the "keys empty at import" bug.
_supabase = None

def _get_supabase():
    """Lazy-init Supabase client so .env is loaded before we read keys."""
    global _supabase
    if _supabase is not None:
        return _supabase
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_ANON_KEY", "")
    if url and key:
        try:
            from supabase import create_client
            _supabase = create_client(url, key)
        except Exception:
            _supabase = False   # mark as "tried and failed"
    else:
        _supabase = False
    return _supabase

_LOCAL_STORE = Path("uploaded_chats")


def _file_id(raw_bytes: bytes) -> str:
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def save_chat(raw_bytes: bytes, filename: str) -> dict:
    fid  = _file_id(raw_bytes)
    dest = f"{fid}_{filename}"

    client = _get_supabase()
    if client:
        try:
            client.storage.from_("whatsapp-chats").upload(
                dest, raw_bytes,
                file_options={"content-type": "text/plain"}
            )
            return {"storage": "supabase", "path": dest, "file_id": fid}
        except Exception:
            pass

    # Local fallback
    _LOCAL_STORE.mkdir(exist_ok=True)
    (_LOCAL_STORE / dest).write_bytes(raw_bytes)    # ← was missing underscore bug
    return {"storage": "local", "path": str(_LOCAL_STORE / dest), "file_id": fid}


# ── Stop-words ────────────────────────────────────────────────────────────────

_STOP_WORDS: set = set()

def _load_stop_words() -> set:
    global _STOP_WORDS
    if _STOP_WORDS:
        return _STOP_WORDS
    for candidate in ["stop_hinglish.txt", "stop_words.txt"]:
        p = Path(candidate)
        if p.exists():
            _STOP_WORDS = set(p.read_text(encoding="utf-8").split())
            return _STOP_WORDS
    _STOP_WORDS = {
        "the","a","an","is","it","in","on","at","to","of","and","or",
        "for","with","this","that","was","are","be","have","has","had",
        "i","me","my","we","our","you","your","he","she","they","them",
        "his","her","their","will","would","could","should","do","did",
        "not","no","yes","ok","okay","hi","hello","hey","lol","haha",
        "just","like","so","but","if","from","by","about","its",
        "what","when","where","who","how","why","there","here","time",
        "media","omitted","deleted","edited","message","pm","am",
    }
    return _STOP_WORDS


# ── Core statistics ───────────────────────────────────────────────────────────

def fetch_stats(selected_user: str, df: pd.DataFrame) -> dict:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]

    if df.empty:
        return {"num_messages":0,"num_words":0,"num_media":0,
                "num_links":0,"avg_msg_len":0.0,"total_emojis":0,"top_emoji":"—"}

    num_messages = df.shape[0]
    num_words    = int(df["message"].apply(lambda m: len(m.split())).sum())
    num_media    = int(df["message"].str.contains("Media omitted", case=False, na=False).sum())
    num_links    = int(df["message"].str.contains(r"https?://", regex=True, na=False).sum())
    avg_msg_len  = round(df["message"].apply(lambda m: len(m.split())).mean() or 0, 1)

    all_emojis   = [ch for msg in df["message"] for ch in msg if ch in emoji.EMOJI_DATA]
    total_emojis = len(all_emojis)
    top_emoji    = Counter(all_emojis).most_common(1)[0][0] if all_emojis else "—"

    return {
        "num_messages": num_messages,
        "num_words":    num_words,
        "num_media":    num_media,
        "num_links":    num_links,
        "avg_msg_len":  avg_msg_len,
        "total_emojis": total_emojis,
        "top_emoji":    top_emoji,
    }


# ── User activity ─────────────────────────────────────────────────────────────

def most_busy_users(df: pd.DataFrame):
    counts = df["user"].value_counts().head(10)
    pct    = df["user"].value_counts(normalize=True).mul(100).round(2).reset_index()
    # pandas <2.0 → columns: index, user  |  pandas ≥2.0 → columns: user, proportion
    pct.columns = ["name", "percent"]
    return counts, pct


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
    return pd.DataFrame(Counter(all_emojis).most_common(n), columns=["emoji", "count"])


# ── Timelines ─────────────────────────────────────────────────────────────────

def monthly_timeline(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    tl = (
        df.groupby(["year", "month_num", "month"])
        .size().reset_index(name="message")
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
    return df["day_name"].value_counts().reindex(_DAY_ORDER, fill_value=0)


def month_activity_map(selected_user: str, df: pd.DataFrame) -> pd.Series:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    return df["month"].value_counts().reindex(_MONTH_ORDER, fill_value=0).dropna()


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
    d = df[df["user"] != "group_notification"].copy().sort_values("date")
    d["prev_user"] = d["user"].shift(1)
    d["prev_time"] = d["date"].shift(1)
    d["gap_min"]   = (d["date"] - d["prev_time"]).dt.total_seconds() / 60

    replies = d[(d["user"] != d["prev_user"]) & (d["gap_min"] < 1440)]
    if replies.empty:
        return pd.DataFrame(columns=["user", "avg_response_min"])
    return (
        replies.groupby("user")["gap_min"]
        .mean().round(1).reset_index()
        .rename(columns={"gap_min": "avg_response_min"})
        .sort_values("avg_response_min")
    )


# ── Sentiment ─────────────────────────────────────────────────────────────────

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

def sentiment_over_time(selected_user: str, df: pd.DataFrame) -> pd.DataFrame:
    if selected_user != "Overall":
        df = df[df["user"] == selected_user]
    temp = df[df["user"] != "group_notification"].copy()

    def _score(msg: str) -> float:
        tokens = set(re.sub(r"[^a-zA-Z\s]", " ", msg).lower().split()) | set(msg)
        return len(tokens & _POS) - len(tokens & _NEG)

    temp["sentiment"] = temp["message"].apply(_score)
    monthly = (
        temp.groupby(["year", "month_num", "month"])["sentiment"]
        .mean().round(2).reset_index()
    )
    monthly["time"] = monthly["month"] + "-" + monthly["year"].astype(str)
    return monthly


# ── AI helpers ────────────────────────────────────────────────────────────────

def _sample_messages(df: pd.DataFrame, max_messages: int = 300) -> list:
    real = df[
        (df["user"] != "group_notification") &
        (~df["message"].str.contains("Media omitted|deleted|edited", case=False, na=False))
    ].copy()

    if len(real) > max_messages:
        step    = max(1, len(real) // max_messages)
        indices = list(range(0, len(real), step))[:max_messages]
        real    = real.iloc[indices]

    return [
        {"user": row["user"], "message": row["message"].strip(), "date": str(row["only_date"])}
        for _, row in real.iterrows()
    ]


def build_chat_context(df: pd.DataFrame) -> str:
    samples = _sample_messages(df)
    return "\n".join(f"[{s['date']}] {s['user']}: {s['message']}" for s in samples)


def compute_relationship_signals(df: pd.DataFrame) -> dict:
    real = df[
        (df["user"] != "group_notification") &
        (~df["message"].str.contains("Media omitted|deleted|edited", case=False, na=False))
    ].copy().sort_values("date")

    users = list(real["user"].unique())
    total = len(real)

    if total == 0:
        return {
            "users": users, "message_share": {}, "reply_pairs": {},
            "endearments_per_user": {}, "question_share": {},
            "late_night_share": {}, "avg_message_length": {},
            "date_range": "N/A", "total_messages": 0, "num_participants": 0,
        }

    # Message share
    share = (real["user"].value_counts() / total * 100).round(1).to_dict()

    # Reply patterns
    real = real.copy()
    real["prev_user"] = real["user"].shift(1)
    real["gap_min"]   = (real["date"] - real["date"].shift(1)).dt.total_seconds() / 60
    replies = real[(real["user"] != real["prev_user"]) & (real["gap_min"] < 60)]
    reply_counts = {}
    for _, row in replies.iterrows():
        pair = (row["prev_user"], row["user"])
        reply_counts[pair] = reply_counts.get(pair, 0) + 1

    # Endearments
    ENDEARMENTS = {
        "babe","baby","love","dear","sweetheart","honey","darling","cutie",
        "jaan","yaar","bro","bhai","dude","sis","bestie","buddy","bff",
        "sir","boss","chief","team","guys","everyone","all",
    }
    endear_hits = {u: Counter() for u in users}
    for _, row in real.iterrows():
        tokens = set(re.sub(r"[^a-z\s]", " ", row["message"].lower()).split())
        for h in tokens & ENDEARMENTS:
            endear_hits[row["user"]][h] += 1

    # Question share — safe division
    real["is_question"] = real["message"].str.contains(r"\?", na=False)
    total_q = real["is_question"].sum()
    if total_q > 0:
        q_share = (real.groupby("user")["is_question"].sum() / total_q * 100).round(1).to_dict()
    else:
        q_share = {u: 0.0 for u in users}

    # Late-night share
    late = real[real["hour"].between(22, 23) | real["hour"].between(0, 4)]
    late_share = (late["user"].value_counts() / len(late) * 100).round(1).to_dict() if len(late) else {}

    # Avg message length
    avg_len = (
        real.groupby("user")["message"]
        .apply(lambda x: round(x.str.split().apply(len).mean(), 1))
        .to_dict()
    )

    return {
        "users":               users,
        "message_share":       share,
        "reply_pairs":         {f"{a}→{b}": c for (a, b), c in sorted(reply_counts.items(), key=lambda x: -x[1])},
        "endearments_per_user":{u: dict(c.most_common(5)) for u, c in endear_hits.items()},
        "question_share":      q_share,
        "late_night_share":    late_share,
        "avg_message_length":  avg_len,
        "date_range":          f"{df['only_date'].min()} to {df['only_date'].max()}",
        "total_messages":      total,
        "num_participants":    len(users),
    }