"""
main.py — WhatsApp Chat Analyzer (Streamlit)
Run:  streamlit run main.py
"""

# ── load .env FIRST before any os.getenv calls ───────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # python-dotenv not installed — fine on Streamlit Cloud

import os
import json
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import preprocessor
import helper

# ── Groq API ──────────────────────────────────────────────────────────────────
_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"


def _get_groq_key() -> str | None:
    try:
        k = st.secrets.get("GROQ_API_KEY", "")
        if k:
            return k
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")  or None


def _call_llm(system: str, user_prompt: str, max_tokens: int = 1800) -> str:
    api_key = _get_groq_key()
    if not api_key:
        return "__NO_KEY__"
    try:
        resp = requests.post(
            _GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": _GROQ_MODEL,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__{e}"


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="WA Chat Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
div[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 18px 20px; transition: transform .15s, border-color .15s;
}
div[data-testid="metric-container"]:hover { transform: translateY(-3px); border-color: #58a6ff; }
div[data-testid="metric-container"] label { color: #8b949e !important; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; }
div[data-testid="metric-container"] [data-testid="metric-value"] { color: #58a6ff !important; font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; }
h1,h2,h3 { font-family:'Syne',sans-serif !important; }
.section-title {
    font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#e6edf3;
    border-left:4px solid #58a6ff; padding-left:12px; margin:2rem 0 1rem;
}
button[data-baseweb="tab"] {
    background: transparent !important; color: #8b949e !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }
.stButton > button {
    background: #238636; color: #fff; border: none; border-radius: 8px;
    padding: 10px 24px; font-family: 'Syne', sans-serif; font-weight: 700;
    letter-spacing: .04em; transition: background .2s; width: 100%;
}
.stButton > button:hover { background: #2ea043; }
[data-testid="stFileUploader"] { background: #161b22; border: 2px dashed #30363d; border-radius: 12px; padding: 12px; }
.dataframe { background: #161b22 !important; color: #e6edf3 !important; }
hr { border-color: #30363d; }
.stAlert { border-radius:10px; }
.hero { text-align: center; padding: 60px 20px 40px; }
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: clamp(2.2rem, 6vw, 4rem); font-weight: 800;
    background: linear-gradient(90deg,#58a6ff,#3fb950,#f78166);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 0.4rem;
}
.hero p { color: #8b949e; font-size:1.05rem; max-width:480px; margin:0 auto; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DARK FIGURE HELPER
# ─────────────────────────────────────────────
BG    = "#0d1117"
CARD  = "#161b22"
BORDER= "#30363d"
BLUE  = "#58a6ff"
GREEN = "#3fb950"
RED   = "#f78166"
GOLD  = "#d29922"
TEXT  = "#e6edf3"
MUTED = "#8b949e"

def _dark_fig(w=10, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.title.set_color(TEXT)
    return fig, ax

# ─────────────────────────────────────────────
# SESSION STATE — persists AI results across reruns
# ─────────────────────────────────────────────
for _key in ["summary_result", "relationship_result", "signals_cache", "analysis_run"]:
    if _key not in st.session_state:
        st.session_state[_key] = None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💬 WA Analyzer")
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Upload exported chat (.txt)",
        type=["txt"],
        help="Export from WhatsApp → More options → Export chat → Without media",
    )

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="hero">
        <h1>WhatsApp Chat Analyzer</h1>
        <p>Upload any WhatsApp export to uncover hidden patterns, stats, and stories in your conversations.</p>
    </div>
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "📊", "Rich Statistics", "Messages, words, media, links & more"),
        (c2, "🕒", "Time Patterns", "When are you most active? Heatmaps reveal all"),
        (c3, "🤖", "AI Insights", "Chat summary + relationship analysis powered by Groq"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;text-align:center;height:140px">
                <div style="font-size:2rem">{icon}</div>
                <div style="font-family:'Syne',sans-serif;font-weight:700;color:#e6edf3;margin:.4rem 0">{title}</div>
                <div style="color:#8b949e;font-size:.85rem">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# LOAD & PARSE
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Parsing chat…")
def load_df(raw_bytes: bytes) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "ISO-8859-1", "cp1252"]:
        try:
            return preprocessor.preprocess(raw_bytes.decode(enc))
        except (UnicodeDecodeError, ValueError):
            continue
    raise ValueError("Unable to decode or parse this file. Please check the format.")


raw_bytes = uploaded_file.getvalue()

# ── Optional cloud save ──
try:
    save_result = helper.save_chat(raw_bytes, uploaded_file.name)
    if save_result["storage"] == "supabase":
        st.sidebar.success("☁️ Chat is readed")
    else:
        st.sidebar.info("💾 Chat is readed ")
except Exception:
    pass

try:
    df = load_df(raw_bytes)
except Exception as e:
    st.error(f"❌ Could not parse chat: {e}")
    st.info("Make sure you exported the chat as **plain text (without media)** from WhatsApp.")
    st.stop()

# Clear AI cache if a new file is uploaded
if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded_file.name:
    st.session_state["last_file"]           = uploaded_file.name
    st.session_state["summary_result"]      = None
    st.session_state["relationship_result"] = None
    st.session_state["signals_cache"]       = None

# ── User selector ──
user_list = [u for u in df["user"].unique() if u != "group_notification"]
user_list.sort()
user_list.insert(0, "Overall")

with st.sidebar:
    selected_user = st.selectbox("Analyse for", user_list)
    run = st.button("🚀 Show Analysis")
    st.markdown("---")
    st.caption(f"📁 {uploaded_file.name}")
    st.caption(f"📨 {df.shape[0]:,} total messages")
    date_range = f"{df['only_date'].min()} → {df['only_date'].max()}"
    st.caption(f"📅 {date_range}")

    # ── Supabase status check ──
    st.markdown("---")
    st.markdown("**🗄 Storage Status**")
    try:
        supa_url = st.secrets.get("SUPABASE_URL", "") or os.getenv("SUPABASE_URL", "")
        supa_key = st.secrets.get("SUPABASE_ANON_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
        if supa_url and supa_key:
            # Ping Supabase with a lightweight health check
            r = requests.get(
                f"{supa_url}/storage/v1/bucket",
                headers={"apikey": supa_key, "Authorization": f"Bearer {supa_key}"},
                timeout=5,
            )
            if r.status_code == 200:   # 404 = connected but no tables, still means auth works
                st.success("☁️ Supabase connected")
            else:
                st.warning(f"⚠️ Supabase returned {r.status_code}")
        else:
            st.caption("Supabase not configured\n(chats save locally)")
    except Exception as ex:
        st.warning(f"⚠️ Supabase unreachable\n{ex}")

    # ── Groq status check ──
    groq_ok = bool(_get_groq_key())
    if groq_ok:
        st.success("🤖 Groq AI ready")
    else:
        st.caption("Groq key not set\n(AI tab disabled)")

if run:
    st.session_state["analysis_run"] = True

if not st.session_state.get("analysis_run"):
    st.info("👈 Select a user and click **Show Analysis** to get started.")
    st.stop()

# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────
st.markdown(
    f"<h1 style='font-family:Syne,sans-serif;margin-bottom:.2rem'>Analysis — {selected_user}</h1>",
    unsafe_allow_html=True,
)
st.markdown(f"<p style='color:#8b949e'>WhatsApp export · {date_range}</p>", unsafe_allow_html=True)
st.markdown("---")

stats = helper.fetch_stats(selected_user, df)

cols = st.columns(7)
kpis = [
    ("Messages",     stats["num_messages"],  ""),
    ("Words",        stats["num_words"],     ""),
    ("Media Shared", stats["num_media"],     ""),
    ("Links",        stats["num_links"],     ""),
    ("Avg Msg Len",  stats["avg_msg_len"],   " words"),
    ("Emojis Used",  stats["total_emojis"],  ""),
    ("Top Emoji",    stats["top_emoji"],     ""),
]
for col, (label, value, suffix) in zip(cols, kpis):
    col.metric(label, f"{value}{suffix}")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_timeline, tab_activity, tab_words, tab_emoji, tab_users, tab_sentiment, tab_ai = st.tabs([
    "📈 Timeline", "🗓 Activity", "☁️ Words", "😀 Emojis", "👥 Users", "💬 Sentiment", "🤖 AI Insights"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — TIMELINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_timeline:
    st.markdown('<div class="section-title">Monthly Message Volume</div>', unsafe_allow_html=True)
    tl = helper.monthly_timeline(selected_user, df)
    fig, ax = _dark_fig(12, 4)
    ax.plot(tl["time"], tl["message"], color=BLUE, lw=2.5, marker="o", markersize=5)
    ax.fill_between(tl["time"], tl["message"], alpha=0.12, color=BLUE)
    ax.set_ylabel("Messages")
    plt.xticks(rotation=45, ha="right", color=MUTED, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-title">Daily Message Volume</div>', unsafe_allow_html=True)
    dt = helper.daily_timeline(selected_user, df)
    fig, ax = _dark_fig(12, 4)
    ax.plot(dt["only_date"], dt["message"], color=GREEN, lw=1.5, alpha=0.9)
    ax.fill_between(dt["only_date"], dt["message"], alpha=0.1, color=GREEN)
    ax.set_ylabel("Messages")
    plt.xticks(rotation=45, ha="right", color=MUTED, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — ACTIVITY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_activity:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Busiest Day of Week</div>', unsafe_allow_html=True)
        busy_day = helper.week_activity_map(selected_user, df)
        fig, ax = _dark_fig(6, 4)
        ax.bar(busy_day.index, busy_day.values,
               color=[BLUE if v == busy_day.max() else "#21262d" for v in busy_day.values],
               edgecolor=BORDER, linewidth=.8)
        ax.set_ylabel("Messages")
        plt.xticks(rotation=30, ha="right", color=MUTED, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        st.markdown('<div class="section-title">Busiest Month</div>', unsafe_allow_html=True)
        busy_month = helper.month_activity_map(selected_user, df)
        fig, ax = _dark_fig(6, 4)
        ax.bar(busy_month.index, busy_month.values,
               color=[GOLD if v == busy_month.max() else "#21262d" for v in busy_month.values],
               edgecolor=BORDER, linewidth=.8)
        ax.set_ylabel("Messages")
        plt.xticks(rotation=30, ha="right", color=MUTED, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-title">Weekly Activity Heatmap</div>', unsafe_allow_html=True)
    heatmap_df = helper.activity_heatmap(selected_user, df)
    if not heatmap_df.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)
        sns.heatmap(heatmap_df, ax=ax, cmap="YlOrRd", linewidths=.4, linecolor=BG, cbar_kws={"shrink": .6})
        ax.tick_params(colors=MUTED, labelsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="section-title">Average Response Time (minutes)</div>', unsafe_allow_html=True)
    rt = helper.response_time_analysis(df)
    if not rt.empty:
        fig, ax = _dark_fig(8, 4)
        colors = [GREEN if i == 0 else "#21262d" for i in range(len(rt))]
        ax.barh(rt["user"], rt["avg_response_min"], color=colors, edgecolor=BORDER)
        ax.set_xlabel("Minutes")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("Not enough data to compute response times.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — WORDS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_words:
    st.markdown('<div class="section-title">Word Cloud</div>', unsafe_allow_html=True)
    try:
        wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor(CARD)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"Could not generate word cloud: {e}")

    st.markdown('<div class="section-title">Top 20 Most-Used Words</div>', unsafe_allow_html=True)
    common = helper.most_common_words(selected_user, df, n=20)
    if not common.empty:
        fig, ax = _dark_fig(10, 5)
        palette = [BLUE] + ["#21262d"] * (len(common) - 1)
        ax.bar(common["word"], common["count"], color=palette, edgecolor=BORDER)
        ax.set_ylabel("Frequency")
        plt.xticks(rotation=40, ha="right", color=MUTED, fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.info("No common words found.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — EMOJIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_emoji:
    st.markdown('<div class="section-title">Top Emojis Used</div>', unsafe_allow_html=True)
    emoji_df = helper.emoji_analysis(selected_user, df)
    if emoji_df.empty:
        st.info("No emojis found.")
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig, ax = _dark_fig(8, 5)
            ax.bar(emoji_df["emoji"], emoji_df["count"],
                   color=[RED, GOLD, BLUE, GREEN, MUTED] * 5, edgecolor=BORDER)
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with c2:
            st.dataframe(
                emoji_df.style.background_gradient(subset=["count"], cmap="Blues"),
                use_container_width=True, height=420,
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — USERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_users:
    if selected_user != "Overall":
        st.info("Switch to **Overall** to see per-user comparisons.")
    else:
        st.markdown('<div class="section-title">Most Active Users</div>', unsafe_allow_html=True)
        counts, pct_df = helper.most_busy_users(df)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = _dark_fig(8, 5)
            colors = [RED if i == 0 else "#21262d" for i in range(len(counts))]
            ax.bar(counts.index, counts.values, color=colors, edgecolor=BORDER)
            ax.set_ylabel("Messages")
            plt.xticks(rotation=30, ha="right", color=MUTED, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        with c2:
            st.dataframe(pct_df, use_container_width=True, height=350)

        st.markdown('<div class="section-title">Message Share</div>', unsafe_allow_html=True)
        top10 = counts.head(8).copy()
        others = counts.iloc[8:].sum()
        if others > 0:
            top10["Others"] = others
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor(CARD)
        wedge_colors = [BLUE, GREEN, RED, GOLD, MUTED, "#c9d1d9", "#388bfd", "#56d364", "#ff7b72"]
        ax.pie(
            top10.values, labels=top10.index, autopct="%1.1f%%", startangle=140,
            colors=wedge_colors[:len(top10)],
            textprops={"color": TEXT, "fontsize": 9},
            wedgeprops={"edgecolor": CARD, "linewidth": 2},
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — SENTIMENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_sentiment:
    st.markdown('<div class="section-title">Conversation Mood Over Time</div>', unsafe_allow_html=True)
    st.caption("Positive score = more happy words; negative = more negative words (lexicon-based).")
    sent = helper.sentiment_over_time(selected_user, df)
    if not sent.empty:
        fig, ax = _dark_fig(12, 4)
        colors_sent = [GREEN if s >= 0 else RED for s in sent["sentiment"]]
        ax.bar(sent["time"], sent["sentiment"], color=colors_sent, edgecolor=BORDER, width=0.8)
        ax.axhline(0, color=MUTED, linewidth=0.8, linestyle="--")
        ax.set_ylabel("Avg Sentiment Score")
        plt.xticks(rotation=45, ha="right", color=MUTED, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        overall_mood = sent["sentiment"].mean()
        mood_label = "😊 Generally Positive" if overall_mood > 0 else ("😐 Neutral" if overall_mood == 0 else "😟 Generally Negative")
        st.metric("Overall Mood", mood_label, f"{overall_mood:+.2f}")
    else:
        st.info("Not enough data to compute sentiment.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 7 — AI INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_ai:
    st.markdown('<div class="section-title">🤖 AI-Powered Chat Insights</div>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e'>Groq (free LLM) reads a sample of your chat and gives you a smart "
        "summary plus relationship analysis.</p>",
        unsafe_allow_html=True,
    )

    api_key_present = bool(_get_groq_key())

    if not api_key_present:
        st.warning(
            "**Groq API key not found.**\n\n"
            "**Get free key:** [console.groq.com](https://console.groq.com) → API Keys → Create\n\n"
            "**Local** — add to `.env` file:\n```\nGROQ_API_KEY=gsk_...\n```\n\n"
            "**Streamlit Cloud** — App → ⚙️ Settings → Secrets:\n```toml\nGROQ_API_KEY = \"gsk_...\"\n```",
            icon="🔑",
        )

    # ── Two buttons side by side ──────────────────────────────────────────────
    # IMPORTANT: buttons are OUTSIDE columns — this prevents the page-jump bug.
    # Results are stored in session_state so they survive the rerun.

    b1, b2 = st.columns(2)
    with b1:
        if st.button("✨ Generate Summary", disabled=not api_key_present, key="btn_summary"):
            with st.spinner("Reading through the chat…"):
                chat_text = helper.build_chat_context(df)
                signals   = helper.compute_relationship_signals(df)
                system = (
                    "You are an expert conversation analyst. Analyse these WhatsApp messages "
                    "(may be English, Hindi, Hinglish, or mixed). Be warm, specific, and insightful. "
                    "Use bullet points with emojis. Keep under 300 words."
                )
                user_prompt = (
                    f"Sample of {signals['total_messages']} messages between "
                    f"{signals['num_participants']} people ({signals['date_range']}).\n\n"
                    f"CHAT SAMPLE:\n{chat_text}\n\n"
                    "Provide:\n"
                    "1. **What this chat is about** — main topics and themes\n"
                    "2. **Notable moments** — jokes, plans, arguments, celebrations\n"
                    "3. **Communication style** — formal/casual, humour, language mix\n"
                    "4. **Interesting observations** about how they talk\n"
                )
                st.session_state["summary_result"] = _call_llm(system, user_prompt, max_tokens=600)

    with b2:
        if st.button("🔍 Analyse Relationships", disabled=not api_key_present, key="btn_rel"):
            with st.spinner("Figuring out the vibe…"):
                chat_text = helper.build_chat_context(df)
                signals   = helper.compute_relationship_signals(df)
                sig_text  = json.dumps({
                    "participants": signals["users"],
                    "message_share_pct": signals["message_share"],
                    "fast_reply_pairs": dict(list(signals["reply_pairs"].items())[:10]),
                    "terms_of_endearment": signals["endearments_per_user"],
                    "question_share_pct": signals["question_share"],
                    "late_night_share_pct": signals["late_night_share"],
                    "avg_words_per_msg": signals["avg_message_length"],
                }, indent=2, ensure_ascii=False)
                system = (
                    "You are a social dynamics expert. Analyse WhatsApp conversations to understand "
                    "relationships. Be specific, name participants, be warm and occasionally witty."
                )
                user_prompt = (
                    f"Analyse the relationship(s) in this chat.\n\n"
                    f"SIGNALS (hard data):\n{sig_text}\n\n"
                    f"CHAT SAMPLE ({signals['total_messages']} messages, {signals['date_range']}):\n{chat_text}\n\n"
                    "For each pair or the group, explain:\n"
                    "1. **Relationship Type** — couple, best friends, siblings, coworkers, classmates, etc.\n"
                    "2. **Confidence & Evidence** — what signals led you here?\n"
                    "3. **Relationship Dynamics** — who initiates, who's funnier, power dynamics?\n"
                    "4. **Bond Strength** — Acquaintances / Casual / Good Friends / Very Close / Inseparable\n"
                    "5. **Fun Observation** — one surprising or heartwarming thing\n"
                )
                st.session_state["relationship_result"] = _call_llm(system, user_prompt, max_tokens=900)
                st.session_state["signals_cache"] = signals

    # ── Display results (persisted in session_state — no page jump) ──────────
    r_sum = st.session_state["summary_result"]
    r_rel = st.session_state["relationship_result"]

    if r_sum or r_rel:
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            if r_sum:
                st.markdown("#### 📋 Chat Summary")
                if r_sum == "__NO_KEY__":
                    st.error("API key missing.")
                elif r_sum.startswith("__ERROR__"):
                    st.error(f"Error: {r_sum[9:]}")
                else:
                    st.markdown(
                        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
                        f"padding:20px;font-size:.92rem;line-height:1.7'>{r_sum}</div>",
                        unsafe_allow_html=True,
                    )

        with res_col2:
            if r_rel:
                st.markdown("#### 💞 Relationship Analysis")
                if r_rel == "__NO_KEY__":
                    st.error("API key missing.")
                elif r_rel.startswith("__ERROR__"):
                    st.error(f"Error: {r_rel[9:]}")
                else:
                    st.markdown(
                        f"<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
                        f"padding:20px;font-size:.92rem;line-height:1.7'>{r_rel}</div>",
                        unsafe_allow_html=True,
                    )

    # ── Raw signals expander ──────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔬 View raw relationship signals (what the AI sees)"):
        sig = st.session_state.get("signals_cache") or helper.compute_relationship_signals(df)
        st.json(sig)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#30363d;font-size:.8rem'>"
    "WhatsApp Chat Analyzer · Built with Streamlit · Your data stays on your machine"
    "</p>",
    unsafe_allow_html=True,
)