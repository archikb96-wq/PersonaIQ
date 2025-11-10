# app.py — PersonaIQ: Reviews → Personas → LLM Copy (final, canonical reports)

import os, re, json, requests
import numpy as np
import pandas as pd
import streamlit as st

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from google_play_scraper import reviews as gp_reviews, Sort

# ---------------- Page setup ----------------
st.set_page_config(page_title="PersonaIQ: Reviews → Personas", layout="wide")
st.title("PersonaIQ: Convert User Reviews into Personas and Generate Targeted Campaigns")

st.caption(
    "Upload a CSV with a **review** column, or paste a **Play Store App ID / URL** "
    "to fetch reviews. Pipeline: Sentiment → Topics → Personas → (optional) Ollama rewrite → Persona reports."
)

# ---------------- Helpers ----------------
def extract_app_id(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    if "http" not in text and " " not in text and "." in text:
        return text
    m = re.search(r"[?&]id=([A-Za-z0-9._]+)", text)
    return m.group(1) if m else ""

def compound_to_label(c: float) -> str:
    if c > 0.7:
        return "Highly Positive"
    if 0.2 < c <= 0.7:
        return "Mildly Positive"
    if -0.2 <= c <= 0.2:
        return "Neutral"
    return "Negative"

def craft_copy(words, sentiment):
    """Generate marketing copy based on 4 sentiment tiers."""
    key1 = words[-1] if words else "service"
    key2 = words[-2] if len(words) > 1 else "experience"

    if sentiment > 0.7:
        tone = "enthusiastic"
        pain = f"Minor {key2} issues occasionally faced"
        need = f"Continue excellent {key1} quality"
        headline = f"Love Your {key1.title()} More"
        body = f"Our users adore the {key1} experience. Consistent, joyful, and always improving!"
        cta = "Share your joy"
    elif 0.2 < sentiment <= 0.7:
        tone = "optimistic"
        pain = f"Some {key2} improvements expected"
        need = f"Better {key1} reliability"
        headline = f"Trust Your {key1.title()}"
        body = f"Enjoy dependable {key1} and smoother {key2} every time — made for your peace of mind."
        cta = "Experience more"
    elif -0.2 <= sentiment <= 0.2:
        tone = "practical"
        pain = f"Inconsistent {key1} or {key2}"
        need = f"Reliable {key1} performance"
        headline = f"Make {key1.title()} Reliable"
        body = f"Get steady {key1} and smoother {key2}. Simple, dependable, every time."
        cta = "Try now"
    else:
        tone = "reassuring"
        pain = f"Frequent issues with {key1}"
        need = f"Fix {key1} and improve {key2}"
        headline = f"Goodbye {key1.title()} Hassles"
        body = f"We’re improving {key1} and {key2} with faster resolution and better support."
        cta = "See updates"

    return dict(tone=tone, pain=pain, need=need, headline=headline, body=body, cta=cta)

def local_rewrite(row, voice="Trustworthy"):
    """Offline tightening if Ollama not used/available."""
    h = row["headline"][:30].strip()
    b = row["body"].replace("  ", " ").strip()[:90]
    c = row["cta"][:20].strip()
    if voice.lower() == "premium":
        b = b.replace("faster", "swift").replace("easy", "effortless")
    elif voice.lower() == "youthful":
        b = b.replace("simple", "super simple").replace("reliable", "super reliable")
    return {"gpt_headline": h, "gpt_text": b, "gpt_cta": c}

def ollama_rewrite(row, model="phi3", brand_voice="Trustworthy", discount_cap="10%"):
    """Local LLM rewrite via Ollama. Falls back to local_rewrite on error."""
    prompt = f"""
You are a senior marketing copywriter.
Brand voice: {brand_voice}. Max discount: {discount_cap}.
Keep headline<=30 chars, text<=90 chars, CTA<=20 chars. No emojis.

Persona: {row['name']}
Top words: {row['top_words']}
Sentiment: {row['avg_sentiment']}

Existing copy:
Headline: {row['headline']}
Body: {row['body']}
CTA: {row['cta']}

Rewrite headline, text, and CTA in JSON with keys: headline, text, cta.
"""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        data = resp.json()
        text = (data.get("response") or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            j = json.loads(m.group(0))
            return {
                "gpt_headline": j.get("headline", "")[:30].strip(),
                "gpt_text": j.get("text", "")[:90].strip(),
                "gpt_cta": j.get("cta", "")[:20].strip(),
            }
    except Exception:
        pass
    return local_rewrite(row, voice=brand_voice)

# -------- Canonical persona report helpers (fixed layout) --------
def _strip_fences(text: str) -> str:
    """Remove code fences and extra whitespace from LLM output."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'```.*?```', '', text, flags=re.S)  # remove code blocks
    text = re.sub(r'`{1,3}', '', text)                # stray backticks
    text = re.sub(r'\s+\n', '\n', text)               # trim trailing spaces before newline
    text = re.sub(r'\n{3,}', '\n\n', text)            # collapse extra blank lines
    return text.strip()

def _build_canonical_report(row: pd.Series, insight: str = "") -> str:
    """Always render the same clean, short report layout, tied to LLM rewritten copy."""
    sent_label = compound_to_label(float(row["avg_sentiment"]))
    headline = row.get("gpt_headline", row.get("headline", "")).strip()
    primary  = row.get("gpt_text", row.get("body", "")).strip()
    cta      = row.get("gpt_cta", row.get("cta", "")).strip()

    # keep insight short (≤ ~220 chars, max ~2 sentences)
    insight = (insight or "").strip()
    if len(insight) > 220:
        insight = insight[:220].rsplit(" ", 1)[0] + "…"

    return f"""### {row['name']}
- **Name:** {row['name']} (Size: {row['size']})
- **Sentiment Score:** {sent_label} ({float(row['avg_sentiment']):+.2f})
- **Key Desires & Pain Points:** {row.get('pain','—')}
- **Top Priorities in Marketing Messages:** Emphasize {row.get('need','—')}.
- **Rewritten Copy:**  
  • **Headline:** _{headline}_  
  • **Primary Text:** {primary}  
  • **CTA:** _{cta}_
- **Marketer Insight:** {insight or "Focus messaging on the clearest need and mirror the tone of the rewritten copy."}
"""

def ollama_persona_report(row, model="phi3", brand_voice="Trustworthy"):
    """
    Ask the LLM for a 1–2 sentence strategic insight, then render a fixed-format report.
    This avoids layout drift while keeping the 'brain' of the LLM.
    """
    prompt = f"""
You are a marketing strategist.
Given the rewritten ad copy below, write ONE or TWO short sentences advising what to say to this persona and why.
Do NOT return headings, lists, code blocks, or JSON. Just the sentences.

Persona name: {row['name']}
Avg sentiment: {row['avg_sentiment']:.2f}
Rewritten copy:
- Headline: "{row.get('gpt_headline', row.get('headline',''))}"
- Text: "{row.get('gpt_text', row.get('body',''))}"
- CTA: "{row.get('gpt_cta', row.get('cta',''))}"
"""
    insight = ""
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        data = resp.json()
        insight = _strip_fences((data.get("response") or ""))
        parts = re.split(r'(?<=[.!?])\s+', insight)
        insight = " ".join(parts[:2]).strip()
    except Exception:
        insight = ""

    return _build_canonical_report(row, insight)

# ---------------- Sidebar (inputs) ----------------
st.sidebar.header("1) Upload Or Fetch Reviews")
uploaded = st.sidebar.file_uploader("Upload CSV (must have a 'review' column)", type=["csv"])

app_id_input = st.sidebar.text_input("Or Paste Play Store App ID / URL (e.g., in.swiggy.android)")
country = st.sidebar.text_input("Country Code", value="in")
n_fetch = st.sidebar.slider("How Many Reviews To Fetch", 100, 1500, 500, step=100)
fetch_btn = st.sidebar.button("Fetch Reviews")

st.sidebar.markdown("---")
st.sidebar.header("2) Modeling Settings")
n_topics = st.sidebar.slider("Personas (Topics)", 3, 8, 5)
max_vocab = st.sidebar.slider("Max Vocab Size", 1000, 30000, 5000, step=1000)
top_terms = st.sidebar.slider("Top Words Per Topic", 5, 15, 8)

st.sidebar.markdown("---")
st.sidebar.header("3) Copy Settings")
brand_voice = st.sidebar.selectbox("Brand Voice", ["Trustworthy","Friendly","Premium","Youthful","Formal"], index=0)
enable_llm = st.sidebar.checkbox("Use Ollama Rewrite", value=False)
ollama_model = st.sidebar.text_input("Ollama Model", value="phi3")
rewrite_top_k = st.sidebar.slider("Rewrite Top K Largest Personas", 1, n_topics, min(3, n_topics))

# ---------------- Data load ----------------
nltk.download("vader_lexicon", quiet=True)
df, source_label = None, ""

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if "review" not in df.columns:
            st.error("Uploaded CSV must contain a 'review' column.")
            st.stop()
        source_label = "CSV Upload"
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
elif fetch_btn and app_id_input.strip():
    app_id = extract_app_id(app_id_input)
    if not app_id:
        st.error("Could not detect a valid App ID. Paste a Play Store URL or an ID like 'in.swiggy.android'.")
        st.stop()
    with st.sidebar.status("Fetching Reviews From Google Play...", expanded=True):
        try:
            result, _ = gp_reviews(
                app_id, lang="en", country=country.strip() or "in",
                sort=Sort.NEWEST, count=int(n_fetch)
            )
            if not result:
                st.error("No reviews returned. Try another app or increase count.")
                st.stop()
            df = pd.DataFrame(result)

            # Keep ALL columns; only rename 'content' → 'review'
            if "content" in df.columns:
                df = df.rename(columns={"content": "review"})

            # Drop only empty reviews; preserve every other column
            df = df.dropna(subset=["review"]).reset_index(drop=True)
            source_label = f"Play Store: {app_id} ({country}), {len(df)} Rows"
        except Exception as e:
            st.error(f"Play Store fetch failed: {e}")
            st.stop()
else:
    st.info("Upload a CSV or paste an App ID/URL to start. Or click to insert a tiny sample dataset.")
    if st.button("Insert Sample Data"):
        df = pd.DataFrame({
            "userName": ["A","B","C","D","E"],
            "score": [5,2,5,1,4],
            "at": pd.to_datetime(["2024-10-01","2024-10-02","2024-10-03","2024-10-04","2024-10-05"]),
            "review":[
                "Loved the quick delivery and clean packaging. App is easy to navigate.",
                "Payment failed twice and no timely support from chat.",
                "Super fast delivery on weekends. Great coupons and accurate ETAs.",
                "App crashes at checkout. Refund took long, no updates.",
                "Dark mode looks great. Search is accurate and suggestions help."
            ],
            "thumbsUpCount":[3,0,8,2,1]
        })
        source_label = "Sample Data"
    else:
        st.stop()

# Normalize text field
df["review"] = df["review"].astype(str).str.strip()
df = df[df["review"].str.len() > 0].reset_index(drop=True)

# ① Data Preview (max 5 rows, review column only for display)
st.subheader("① Data Preview")
st.caption(f"Source: **{source_label}**")
preview = df[["review"]].head(5).rename(columns={"review": "Review"})
st.dataframe(preview, use_container_width=True)

# ---------------- Sentiment (adds a column; do NOT drop other fields) ----------------
sid = SentimentIntensityAnalyzer()
df["sentiment]"] = df["review"].apply(lambda x: sid.polarity_scores(x)["compound"])  # <-- keep exact quote fix below

# Fix accidental bracket typo if it ever happens
if "sentiment" not in df.columns and "sentiment]" in df.columns:
    df = df.rename(columns={"sentiment]": "sentiment"})

# --- Download ALL Play Store fields + sentiment ---
preferred_order = [
    "userName", "score", "at", "review", "thumbsUpCount", "replyContent",
    "reviewCreatedVersion", "appVersion", "replyAt", "repliedAt", "sentiment"
]
ordered = [c for c in preferred_order if c in df.columns]
rest = [c for c in df.columns if c not in ordered]
export_df = df[ordered + rest].copy()

for col in ["at", "replyAt", "repliedAt"]:
    if col in export_df.columns:
        export_df[col] = export_df[col].astype(str)

st.download_button(
    label="Download Reviews + Sentiment (CSV)",
    data=export_df.to_csv(index=False).encode("utf-8"),
    file_name="reviews_with_sentiment.csv",
    mime="text/csv",
    help="Exports all available Play Store fields (name, stars, date, etc.) plus computed sentiment"
)

# ---------------- Topic modeling ----------------
vectorizer = CountVectorizer(stop_words="english", max_features=max_vocab, min_df=2)
X = vectorizer.fit_transform(df["review"])

lda = LDA(n_components=n_topics, random_state=42)
W = lda.fit_transform(X)
H = lda.components_
vocab = np.array(vectorizer.get_feature_names_out())

# Top terms per topic
topic_terms = []
for k in range(n_topics):
    idx = H[k].argsort()[-top_terms:]
    topic_terms.append(vocab[idx].tolist())

# Assign dominant topic per review
df["topic"] = W.argmax(axis=1)
topic_sent = df.groupby("topic")["sentiment"].mean().reindex(range(n_topics)).fillna(0).values
topic_counts = df["topic"].value_counts().reindex(range(n_topics)).fillna(0).astype(int).values

# ② Topics & Sentiment (keep visible; base personas table hidden)
st.subheader("② Topics & Sentiment")
col1, col2 = st.columns(2)
with col1:
    st.write("Top Words Per Topic")
    tt_df = pd.DataFrame({"Topic": list(range(n_topics)), "Top Words": [", ".join(t) for t in topic_terms]})
    st.table(tt_df)
with col2:
    st.write("Topic Sizes & Avg Sentiment")
    ts_df = pd.DataFrame({"Topic": list(range(n_topics)), "Size": topic_counts, "Avg Sentiment": np.round(topic_sent, 3)})
    st.table(ts_df)

# ---------------- Build personas (not displayed) ----------------
rows = []
for k in range(n_topics):
    words = topic_terms[k]
    name = f"{(words[-1] if words else f'Persona {k+1}').title()} Persona"
    copy = craft_copy(words, float(topic_sent[k]))
    rows.append({
        "topic": k,
        "name": name,
        "size": int(topic_counts[k]),
        "avg_sentiment": round(float(topic_sent[k]), 3),
        "top_words": ", ".join(words),
        **copy
    })
personas = pd.DataFrame(rows).sort_values("size", ascending=False)

# ---------------- LLM rewrite (Ollama optional) ----------------
st.subheader("③ LLM Rewritten Copy (Ollama, Optional)")
if enable_llm:
    st.caption("Uses your local Ollama model (e.g., phi3). If not available, falls back to an offline tightening.")

    out = []
    pers_sorted = personas.sort_values("size", ascending=False).reset_index(drop=True)

    for i, r in pers_sorted.iterrows():
        if i < rewrite_top_k:
            rew = ollama_rewrite(r, model=ollama_model, brand_voice=brand_voice)
            is_llm = True
        else:
            rew = local_rewrite(r, voice=brand_voice)
            is_llm = False
        out.append({**r.to_dict(), **rew, "is_llm": is_llm})

    personas_llm = pd.DataFrame(out)

    # Display only marketing-relevant fields (no Topic, no index)
    show_cols = ["name", "gpt_headline", "gpt_text", "gpt_cta"]
    pretty = personas_llm[show_cols].rename(columns={
        "name": "Name", "gpt_headline": "Headline", "gpt_text": "Primary Text", "gpt_cta": "CTA"
    })
    st.dataframe(pretty, use_container_width=True, hide_index=True)

    # ④ Persona Reports (LLM only, canonical layout)
    st.subheader("④ Persona Reports (LLM Only)")
    llm_rows = personas_llm[personas_llm["is_llm"] == True]

    if llm_rows.empty:
        st.info("No LLM-rewritten personas to report. Increase 'Rewrite Top K' or enable LLM.")
    else:
        reports = []
        for _, r in llm_rows.iterrows():
            md = ollama_persona_report(r, model=ollama_model, brand_voice=brand_voice)
            reports.append(md)

        for block in reports:
            with st.container(border=True):
                st.markdown(block)

        st.download_button(
            "Download Persona Reports (Markdown, LLM Only)",
            data=("\n\n---\n\n".join(reports)).encode("utf-8"),
            file_name="persona_reports_llm.md"
        )
else:
    st.info("Enable **Use Ollama Rewrite** in the sidebar to generate AI-polished copy & persona reports.")

# ---------------- Exports ----------------
st.subheader("⑤ Export Base Personas")
st.download_button(
    "Download Base Personas (CSV)",
    data=personas.to_csv(index=False),
    file_name="personas_base.csv"
)
st.caption("Tip: close the CSV in Excel before re-downloading to avoid file lock issues.")
