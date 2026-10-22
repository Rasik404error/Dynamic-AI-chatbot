import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_mic_recorder import speech_to_text

# -------------------- Configuration --------------------
load_dotenv()
st.set_page_config(page_title="Dynamic AI", page_icon="AI", layout="wide")

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "database" / "dynamic_ai.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()

# -------------------- Database --------------------
def connect_db():
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect_db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            assistant_response TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            emotion TEXT NOT NULL,
            response_time REAL,
            feedback INTEGER
        )""")
        con.execute("""CREATE TABLE IF NOT EXISTS preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            preference TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""")

def save_interaction(user, assistant, sentiment, emotion, elapsed):
    with connect_db() as con:
        cur = con.execute(
            """INSERT INTO interactions
            (timestamp,user_message,assistant_response,sentiment,emotion,response_time)
            VALUES (?,?,?,?,?,?)""",
            (pd.Timestamp.now().isoformat(timespec="seconds"), user, assistant,
             sentiment, emotion, float(elapsed)))
        return cur.lastrowid

def save_feedback(row_id, rating):
    with connect_db() as con:
        con.execute("UPDATE interactions SET feedback=? WHERE id=?", (int(rating), int(row_id)))

def get_data():
    with connect_db() as con:
        rows = con.execute("SELECT * FROM interactions ORDER BY timestamp DESC").fetchall()
    df = pd.DataFrame([dict(r) for r in rows])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for col in ["response_time", "feedback"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def get_preferences():
    with connect_db() as con:
        rows = con.execute("SELECT preference FROM preferences ORDER BY id DESC LIMIT 12").fetchall()
    return [r["preference"] for r in rows]

def add_preference(text):
    with connect_db() as con:
        con.execute("INSERT OR IGNORE INTO preferences(preference) VALUES(?)", (text,))

init_db()

# -------------------- Basic sentiment/emotion heuristic --------------------
POS = {"good", "great", "excellent", "awesome", "love", "happy", "helpful", "thanks", "amazing", "nice", "perfect", "fantastic", "excited"}
NEG = {"bad", "terrible", "awful", "hate", "sad", "angry", "upset", "frustrated", "annoyed", "disappointed", "poor", "worst", "confused", "problem", "wrong", "failed"}
EMOTIONS = {
    "happy": ["happy", "glad", "delighted", "pleased"],
    "sad": ["sad", "unhappy", "lonely", "upset"],
    "angry": ["angry", "furious", "mad"],
    "frustrated": ["frustrated", "annoyed", "irritated"],
    "anxious": ["anxious", "worried", "nervous", "scared"],
    "excited": ["excited", "thrilled", "can't wait"],
    "confused": ["confused", "don't understand", "unclear"],
}

def classify_text(text):
    low = text.lower()
    words = set(re.findall(r"[a-z']+", low))
    p, n = len(words & POS), len(words & NEG)
    sentiment = "positive" if p > n else "negative" if n > p else "neutral"
    emotion = next((name for name, terms in EMOTIONS.items()
                    if any(term in low for term in terms)), "neutral")
    if emotion == "neutral" and sentiment == "positive":
        emotion = "happy"
    return sentiment, emotion

# -------------------- OpenRouter API --------------------
def ask_llm(history):
    if not API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing. Add it to .env.")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "Dynamic AI"),
        },
    )
    prefs = get_preferences()
    system = (
        "You are Dynamic AI, a helpful, respectful assistant. "
        "Respond clearly and adapt your tone to the user's message. "
        "Do not claim that text sentiment proves the user's actual emotional state."
    )
    if prefs:
        system += "\nFeedback-based preferences:\n" + "\n".join(f"- {p}" for p in prefs)
    messages = [{"role": "system", "content": system}]
    for m in history[-16:]:
        if m.get("role") in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    result = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.7, max_tokens=1200
    )
    return result.choices[0].message.content or "I couldn't generate a response."

# -------------------- Glass UI --------------------
st.markdown("""
<style>
.stApp {
 background: radial-gradient(ellipse at 8% 10%,rgba(78,70,229,.23),transparent 34%),
 radial-gradient(ellipse at 92% 15%,rgba(217,70,239,.17),transparent 30%),
 radial-gradient(ellipse at 45% 100%,rgba(14,165,233,.13),transparent 38%),#080b18;
 color:#e2e8f0;
}
/* Message composer, microphone, and original arrow-style send button */
div[data-testid="stForm"] {
    border: 0 !important;
    padding: 0 !important;
}
div[data-testid="stForm"] input {
    height: 44px !important;
    min-height: 44px !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,.07) !important;
    border: 1px solid rgba(167,139,250,.32) !important;
    color: #e2e8f0 !important;
}
div[data-testid="stFormSubmitButton"] button {
    height: 44px !important;
    min-height: 44px !important;
    width: 44px !important;
    min-width: 44px !important;
    padding: 0 !important;
    border-radius: 14px !important;
    background: rgba(255,255,255,.07) !important;
    color: #dbeafe !important;
    border: 1px solid rgba(167,139,250,.32) !important;
    font-size: 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background: rgba(139,92,246,.22) !important;
    border-color: rgba(167,139,250,.65) !important;
}
div[data-testid="stHorizontalBlock"] iframe {
    height: 44px !important;
    min-height: 44px !important;
    width: 100% !important;
    display: block !important;
    border: 0 !important;
}
[data-testid="stDataFrame"]{border:1px solid rgba(255,255,255,.1);border-radius:14px;overflow:hidden}
hr{border-color:rgba(255,255,255,.09)}
.footer{text-align:center;color:#64748b;font-size:12px;line-height:1.8;padding:30px 10px 8px}
@media(max-width:768px){.block-container{padding:1rem}.hero{padding:22px}.metric-card{padding:16px;min-height:120px}.metric-value{font-size:23px}.panel{padding:15px}}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    # HTML is intentionally passed as a plain string (not a Markdown code block).
    st.markdown(
        f'<div class="hero"><div class="hero-title">{title}</div>'
        f'<div class="hero-subtitle">{subtitle}</div></div>',
        unsafe_allow_html=True,
    )

def metric(label, value, caption=""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-caption">{caption}</div></div>',
        unsafe_allow_html=True,
    )

def style_chart(fig, height=340):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family":"Arial","color":"#B8C4E0","size":12},
        margin={"l":12,"r":12,"t":52,"b":12}, hovermode="x unified",
        legend={"orientation":"h","y":1.02,"x":1,"xanchor":"right","bgcolor":"rgba(0,0,0,0)"},
        hoverlabel={"bgcolor":"#171D38","bordercolor":"#8B5CF6","font":{"color":"#fff"}},
    )
    fig.update_xaxes(showgrid=False,zeroline=False,linecolor="rgba(148,163,184,.16)",tickfont={"color":"#8795B5"})
    fig.update_yaxes(showgrid=True,gridcolor="rgba(148,163,184,.10)",zeroline=False,tickfont={"color":"#8795B5"})
    return fig

def glow_line(fig, x, y, name, color, fill):
    for width, opacity in [(15,.06),(8,.14)]:
        fig.add_trace(go.Scatter(x=x,y=y,mode="lines",line={"color":color,"width":width,"shape":"spline"},opacity=opacity,hoverinfo="skip",showlegend=False))
    fig.add_trace(go.Scatter(
        x=x,y=y,mode="lines+markers",name=name,
        line={"color":color,"width":3,"shape":"spline"},
        marker={"size":5,"color":color,"line":{"color":"rgba(255,255,255,.7)","width":1}},
        fill="tozeroy",fillcolor=fill,
        hovertemplate="%{x}<br>"+name+": %{y:.2f}<extra></extra>",
    ))
    return fig

def mean_or_none(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if not s.empty else None


# -------------------- Session state --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_interaction_id" not in st.session_state:
    st.session_state.last_interaction_id = None
if "last_feedback_id" not in st.session_state:
    st.session_state.last_feedback_id = None

df = get_data()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 22px">
      <div style="font-size:25px;font-weight:800;color:#fff;letter-spacing:-.5px">Dynamic AI</div>
      <div style="font-size:12px;color:#94a3b8;margin-top:6px">Intelligent Conversation Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("NAVIGATION", ["Chat Assistant", "Analytics Dashboard", "AI Insights"], label_visibility="collapsed")
    st.divider()
    st.markdown("### Session")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_interaction_id = None
        st.session_state.last_feedback_id = None
        st.rerun()
    st.divider()
    st.markdown("### Capabilities")
    st.markdown("- AI-powered conversations\n- Sentiment analysis\n- Emotion classification\n- Feedback-based preferences\n- Interactive analytics")
    st.divider()
    st.caption("Python · Streamlit · OpenRouter")

# -------------------- Chat page --------------------
if page == "Chat Assistant":
    hero("Dynamic AI Assistant", "Intelligent conversations with sentiment-aware responses")
    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Session messages", sum(m["role"]=="user" for m in st.session_state.messages))
    with c2: st.metric("Stored interactions", len(df))
    with c3: st.metric("LLM provider", "OpenRouter")
    st.divider()
    st.subheader("Conversation")
    if not st.session_state.messages:
        st.markdown('<div class="panel"><h3>Welcome to Dynamic AI</h3><p style="color:#94a3b8">Ask a question, explore an idea, or work through a problem.</p></div>', unsafe_allow_html=True)
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"]=="assistant":
                st.caption(f"Sentiment: {m.get('sentiment','neutral').title()} · Emotion classification: {m.get('emotion','neutral').title()}")
                if m.get("response_time") is not None: st.caption(f"Response time: {m['response_time']:.2f} seconds")
    # Transfer a completed voice transcript before creating the input widget.
    if "pending_voice_text" in st.session_state:
        pending = st.session_state.pop("pending_voice_text")
        if pending and pending.strip():
            st.session_state["chat_prompt"] = pending.strip()

    if "chat_prompt" not in st.session_state:
        st.session_state["chat_prompt"] = ""

    # One aligned row: message field + real speech-recognition control + arrow send button.
    input_col, mic_col = st.columns([12, 1], vertical_alignment="center")

    with input_col:
        with st.form("dynamic_ai_chat_form", clear_on_submit=True, border=False):
            field_col, send_col = st.columns([12, 0.8], vertical_alignment="center")
            with field_col:
                typed_prompt = st.text_input(
                    "Message Dynamic AI",
                    key="chat_prompt",
                    placeholder="Type a message or use the microphone...",
                    label_visibility="collapsed",
                )
            with send_col:
                send_clicked = st.form_submit_button(
                    "➤",
                    use_container_width=True,
                    help="Send message (Enter also sends)",
                )

    with mic_col:
        voice_text = speech_to_text(
            language="en",
            start_prompt="🎙",
            stop_prompt="■",
            just_once=True,
            use_container_width=True,
            key="dynamic_ai_speech_input",
        )

    # The component returns text after Stop is clicked. Move it into the input
    # on the next run; don't send automatically, so it can be reviewed/edited.
    if isinstance(voice_text, str) and voice_text.strip():
        transcript = voice_text.strip()
        current_input = st.session_state.get("chat_prompt", "").strip()

        # The recorder may keep returning its last transcript on later reruns.
        # Only transfer it when it is genuinely new; otherwise allow form submission
        # to continue instead of rerunning and interrupting the Send action.
        if transcript != current_input and transcript != st.session_state.get("last_voice_transcript", ""):
            st.session_state["last_voice_transcript"] = transcript
            st.session_state["pending_voice_text"] = transcript
            st.rerun()

    prompt = typed_prompt.strip() if send_clicked else None
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                started = time.perf_counter()
                try:
                    answer = ask_llm(st.session_state.messages)
                    elapsed = time.perf_counter() - started
                    sentiment, emotion = classify_text(prompt)
                    row_id = save_interaction(
                        prompt, answer, sentiment, emotion, elapsed
                    )
                    st.markdown(answer)
                    st.caption(
                        f"Sentiment: {sentiment.title()} · "
                        f"Emotion classification: {emotion.title()} · "
                        f"{elapsed:.2f}s"
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sentiment": sentiment,
                        "emotion": emotion,
                        "response_time": elapsed,
                        "interaction_id": row_id,
                    })
                    st.session_state.last_interaction_id = row_id
                    st.session_state.last_feedback_id = None
                    # clear_on_submit clears the input; retain the transcript guard to avoid replaying stale component output.
                except Exception as exc:
                    st.error(
                        "The AI service could not generate a response. "
                        "Check your .env key, model name, and internet connection."
                    )
                    st.exception(exc)
        st.rerun()

    row_id = st.session_state.last_interaction_id
    if row_id is not None and row_id != st.session_state.last_feedback_id:
        st.divider()
        st.markdown("**Was this response helpful?**")
        f1,f2,_ = st.columns([1,1,4])
        with f1:
            if st.button("Helpful",key=f"good_{row_id}",use_container_width=True):
                save_feedback(row_id,1)
                add_preference("Prefer clear, relevant, helpful responses.")
                st.session_state.last_feedback_id=row_id
                st.success("Feedback saved.")
                st.rerun()
        with f2:
            if st.button("Not helpful",key=f"bad_{row_id}",use_container_width=True):
                save_feedback(row_id,0)
                add_preference("Focus on relevance, clarity, and answering the exact question.")
                st.session_state.last_feedback_id=row_id
                st.success("Feedback saved.")
                st.rerun()

# -------------------- Analytics page --------------------
elif page == "Analytics Dashboard":
    hero("Analytics Overview", "Monitor conversations, sentiment classifications, and response performance")
    if df.empty:
        st.info("No interaction data yet. Start a conversation to populate the dashboard.")
    else:
        filtered = df.copy()
        if filtered["timestamp"].notna().any():
            dates = filtered["timestamp"].dropna()
            start, end = dates.min().date(), dates.max().date()
            selected = st.date_input("Filter date range", value=(start,end), min_value=start, max_value=end)
            if isinstance(selected,(tuple,list)) and len(selected)==2:
                filtered=filtered[filtered["timestamp"].dt.date.between(selected[0],selected[1])].copy()
        if filtered.empty:
            st.warning("No interactions found for this date range.")
        else:
            total=len(filtered)
            positive=int((filtered["sentiment"]=="positive").sum())
            avg=mean_or_none(filtered["response_time"])
            rated=filtered.dropna(subset=["feedback"])
            helpful=float((rated["feedback"]==1).mean()*100) if not rated.empty else None
            st.subheader("Performance Overview")
            a,b,c,d=st.columns(4)
            with a: metric("TOTAL INTERACTIONS",f"{total:,}","Recorded chatbot replies")
            with b: metric("POSITIVE SENTIMENT",f"{positive:,}",f"{positive/total*100:.1f}% of interactions")
            with c: metric("AVG RESPONSE TIME",f"{avg:.2f}s" if avg is not None else "N/A","API request duration")
            with d: metric("HELPFUL RATE",f"{helpful:.1f}%" if helpful is not None else "N/A",f"{len(rated)} rated responses")
            left,right=st.columns(2,gap="large")
            with left:
                st.markdown('<div class="panel">',unsafe_allow_html=True)
                st.subheader("Sentiment Distribution")
                counts=filtered["sentiment"].fillna("unknown").value_counts().rename_axis("Sentiment").reset_index(name="Count")
                fig=px.pie(counts,names="Sentiment",values="Count",hole=.68,color_discrete_sequence=["#A78BFA","#38BDF8","#F472B6","#94A3B8"])
                fig.update_traces(textposition="inside",textinfo="percent+label",marker={"line":{"color":"rgba(255,255,255,.15)","width":2}})
                st.plotly_chart(style_chart(fig,330),use_container_width=True)
                st.markdown("</div>",unsafe_allow_html=True)
            with right:
                st.markdown('<div class="panel">',unsafe_allow_html=True)
                st.subheader("Emotion Distribution")
                counts=filtered["emotion"].fillna("unknown").value_counts().rename_axis("Emotion").reset_index(name="Count")
                fig=px.bar(counts,x="Emotion",y="Count",text="Count",color="Emotion",color_discrete_sequence=["#A78BFA","#38BDF8","#F472B6","#818CF8","#2DD4BF","#FBBF24","#F87171"])
                fig.update_traces(textposition="outside",textfont={"color":"#E2E8F0"})
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_chart(fig,330),use_container_width=True)
                st.markdown("</div>",unsafe_allow_html=True)
            st.markdown('<div class="panel">',unsafe_allow_html=True)
            st.subheader("Daily Chat Activity")
            activity=filtered.dropna(subset=["timestamp"]).copy()
            if not activity.empty:
                activity["Date"]=activity["timestamp"].dt.date
                daily=activity.groupby("Date").size().reset_index(name="Interactions").sort_values("Date")
                fig=go.Figure()
                glow_line(fig,daily["Date"],daily["Interactions"],"Interactions","rgb(168,85,247)","rgba(168,85,247,.12)")
                fig.update_layout(title="Conversation Activity",xaxis_title=None,yaxis_title="Interactions")
                st.plotly_chart(style_chart(fig,370),use_container_width=True)
            else: st.info("No valid timestamps available.")
            st.markdown("</div>",unsafe_allow_html=True)
            st.markdown('<div class="panel">',unsafe_allow_html=True)
            st.subheader("Response Performance")
            perf=filtered.dropna(subset=["timestamp","response_time"]).copy()
            if not perf.empty:
                perf["Date"]=perf["timestamp"].dt.date
                daily=perf.groupby("Date")["response_time"].mean().reset_index().sort_values("Date")
                fig=go.Figure()
                glow_line(fig,daily["Date"],daily["response_time"],"Average response time","rgb(56,189,248)","rgba(56,189,248,.10)")
                fig.update_layout(title="Average API Response Time",xaxis_title=None,yaxis_title="Seconds")
                st.plotly_chart(style_chart(fig,340),use_container_width=True)
            else: st.info("Response-time data will appear after successful interactions.")
            st.markdown("</div>",unsafe_allow_html=True)
            st.markdown('<div class="panel">',unsafe_allow_html=True)
            st.subheader("User Feedback")
            if not rated.empty:
                fc=pd.DataFrame({"Feedback":["Helpful","Not helpful"],"Count":[int((rated["feedback"]==1).sum()),int((rated["feedback"]==0).sum())]})
                fig=px.bar(fc,x="Feedback",y="Count",text="Count",color="Feedback",color_discrete_map={"Helpful":"#2DD4BF","Not helpful":"#F472B6"})
                fig.update_traces(textposition="outside")
                fig.update_layout(showlegend=False)
                st.plotly_chart(style_chart(fig,280),use_container_width=True)
            else: st.info("No feedback has been recorded yet.")
            st.markdown("</div>",unsafe_allow_html=True)
            st.subheader("Recent Interactions")
            columns=[x for x in ["timestamp","user_message","sentiment","emotion","response_time","feedback"] if x in filtered.columns]
            st.dataframe(filtered[columns].head(100),use_container_width=True,hide_index=True)
            st.download_button("Export analytics as CSV",filtered.to_csv(index=False).encode("utf-8"),"dynamic_ai_analytics.csv","text/csv")

# -------------------- Insights page --------------------
elif page == "AI Insights":
    hero("AI Insights","Understand recorded usage patterns and user feedback")
    if df.empty:
        st.info("Start a few conversations to generate insights.")
    else:
        total=len(df)
        pos=float((df["sentiment"]=="positive").mean()*100)
        neg=float((df["sentiment"]=="negative").mean()*100)
        avg=mean_or_none(df["response_time"])
        a,b,c=st.columns(3)
        with a: metric("TOTAL INTERACTIONS",f"{total:,}","Stored chatbot replies")
        with b: metric("POSITIVE SENTIMENT",f"{pos:.1f}%","Rule-based classification")
        with c: metric("NEGATIVE SENTIMENT",f"{neg:.1f}%","Rule-based classification")
        st.subheader("Observations")
        if pos>neg: st.success("Positive sentiment classifications are more common in recorded messages.")
        elif neg>pos: st.warning("Negative sentiment classifications are more common. Review conversations and feedback.")
        else: st.info("Positive and negative sentiment classifications are currently equal.")
        if avg is not None: st.write(f"Average response time: **{avg:.2f} seconds**.")
        rated=df.dropna(subset=["feedback"])
        if not rated.empty:
            st.subheader("Feedback Summary")
            st.metric("Helpful feedback rate",f"{(rated['feedback']==1).mean()*100:.1f}%")
            st.caption(f"Based on {len(rated)} rated responses.")
        emotions=df["emotion"].dropna().value_counts()
        if not emotions.empty:
            st.subheader("Most frequent emotion classification")
            st.write(f"**{emotions.index[0].title()}** — {int(emotions.iloc[0])} interactions")
        st.caption("Sentiment/emotion labels use a lightweight word-based heuristic, not a trained model or clinical assessment. Feedback changes future prompt preferences; it does not retrain LLM weights.")
