import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime
import streamlit.components.v1 as components

# ---------------------------
# Config & User details
# ---------------------------
st.set_page_config(page_title="Spam Classifier Project", page_icon="📩", layout="wide")
AUTHOR = "Chandan Kumar Sah"
CONTACT = "9863981322"
EMAIL = "irisblack0503@gmail.com"

# ---------------------------
# Helpers
# ---------------------------
@st.cache_resource
def load_model(path="spam_clf.pkl"):
    try:
        return joblib.load(path)
    except:
        return None

def interpret_preds(preds):
    out = []
    for x in preds:
        x = str(x).strip().lower()
        if "spam" in x or x == "1":
            out.append("Spam")
        else:
            out.append("Ham")
    return out

def to_bytes_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def to_bytes_text(s):
    return s.encode("utf-8")

def make_readme_bytes():
    txt = f"""Spam Classifier Project

Author: {AUTHOR}
Contact: {CONTACT}
Email: {EMAIL}

Full Web App Features:
• Single & bulk SMS classification
• Exportable CSV results
• Clean Streamlit UI
• Pre-loaded sample datasets
• ML pipeline with TF-IDF + classifier

How to run:
1. Make sure spam_clf.pkl is in the same folder as app.py
2. Run: streamlit run app.py
"""
    return txt.encode("utf-8")

# ---------------------------
# Sample files (expected)
# ---------------------------
LOCAL_SAMPLES = [
    ("spam_ham.txt", "Sample 1 — SMS Spam & Ham Dataset (txt)"),
    ("test_data_reviews.txt", "Sample 2 — Review Messages Dataset (txt)"),
    ("test_data_spam.txt", "Sample 3 — Small Spam Dataset (txt)"),
]

FALLBACK_CONTENT = {
    "spam_ham.txt": "ham\tHello bro how are you?\nspam\tCongratulations! You won ₹5000!",
    "test_data_reviews.txt": "Loved the food.\nService was slow.\nAmazing place.",
    "test_data_spam.txt": "win a free trip now!\nlast chance to win a car\nclick here to claim now",
}

# ensure sample files exist (write fallback only if missing)
for file, _ in LOCAL_SAMPLES:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            f.write(FALLBACK_CONTENT[file])

# ---------------------------
# Load model
# ---------------------------
model = load_model("spam_clf.pkl")

# ---------------------------
# Header HTML (render with components.html)
# ---------------------------
header_html = f"""
<style>
  .app-header {{
    background: linear-gradient(90deg,#ff6a00,#ee0979);
    padding: 28px 22px;
    border-radius: 14px;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    font-family: "Segoe UI", Roboto, Arial;
    overflow: hidden;
  }}
  .app-title {{ font-size: 38px; font-weight: 900; margin-bottom: 8px; }}
  .app-sub {{ font-size: 18px; margin-top: 6px; font-weight: 600; color: rgba(255,255,255,0.98); }}
  .author-badge {{ margin-top: 12px; padding: 9px 18px; background: rgba(255,255,255,0.18); border-radius: 14px; font-size:16px; font-weight:700; display:inline-block; }}
  .pill-links {{ margin-top:12px; display:flex; justify-content:center; gap:12px; }}
  .pill-links a {{ background: rgba(255,255,255,0.24); padding:8px 16px; border-radius:999px; color:white !important; text-decoration:none; font-weight:600; }}
  .pill-links a:hover {{ background: rgba(255,255,255,0.42); }}
  .card {{ padding: 14px; border-radius: 10px; background: white; box-shadow: 0 4px 12px rgba(0,0,0,0.06); }}
  @media(max-width:900px) {{
    .app-title {{ font-size: 28px; }}
    .app-sub {{ font-size: 15px; }}
  }}
</style>

<div class="app-header">
  <div class="app-title">📩 Spam Classifier Project</div>
  <div class="app-sub">💡 <strong>Smart SMS Filtering</strong> · ⚡ <strong>Instant Predictions</strong> · 📤 <strong>Exportable Results</strong></div>
  <div style="height:10px;"></div>
  <div class="author-badge">👨‍💻 Developed by <strong>{AUTHOR}</strong></div>
  <div class="pill-links">
    <a href="https://linkedin.com/in/chandan-kumar-sah-752803387" target="_blank" rel="noopener">🔗 LinkedIn</a>
    <a href="https://github.com/ChankumarSah" target="_blank" rel="noopener">💻 GitHub</a>
  </div>
</div>
"""
components.html(header_html, height=230, scrolling=False)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("## 📌 About this Project")
    st.write("- End-to-end ML pipeline")
    st.write("- Single & bulk prediction")
    st.write("- Exportable results")
    st.markdown("---")

    st.markdown("## 👤 Contact")
    st.write(f"**{AUTHOR}**")
    st.write(f"📞 {CONTACT}")
    st.write(f"📧 {EMAIL}")
    st.markdown("---")

    st.markdown("## 📄 Sample Files (TXT) — download")
    for file, label in LOCAL_SAMPLES:
        try:
            with open(file, "r", encoding="utf-8") as f:
                txt = f.read()
            st.download_button(label=f"📎 {label}", data=to_bytes_text(txt), file_name=file, mime="text/plain")
        except Exception:
            st.write(f"- {label} — not available")

    st.markdown("---")
    st.download_button(label="📘 Download README", data=make_readme_bytes(), file_name="README_spam_classifier.txt", mime="text/plain")

# ---------------------------
# Unified Single + Bulk (single card) — bulk updates in same placeholder
# ---------------------------
st.write("")  # spacing
st.markdown('<div style="max-width:1100px; margin: 0 auto;">', unsafe_allow_html=True)
st.markdown('<div style="padding:18px;border-radius:12px;background:white;box-shadow:0 6px 20px rgba(0,0,0,0.06)">', unsafe_allow_html=True)

st.markdown("### 🔍 Unified Prediction — Single & Bulk", unsafe_allow_html=True)
st.write("Use the text box for a quick single prediction OR upload a CSV/TXT to predict in bulk. Bulk results update in the same place and are downloadable.")

# Single message area
st.markdown("#### ✉️ Single Message Prediction")
single_msg = st.text_area("Enter a single message for quick testing:", height=140, placeholder="Type or paste one SMS/email text here...")
if st.button("Predict Single", key="predict_single"):
    if model is None:
        st.error("Model not loaded — add `spam_clf.pkl` to the project folder.")
    else:
        if not single_msg or not single_msg.strip():
            st.warning("Please enter a message for single prediction.")
        else:
            try:
                single_pred = interpret_preds(model.predict([single_msg]))[0]
                if single_pred == "Spam":
                    st.error("Single Prediction → Spam — Irrelevant / Promotional")
                else:
                    st.success("Single Prediction → Ham — Genuine Message")
            except Exception as e:
                st.error(f"Single prediction error: {e}")

st.write("---")

# Bulk area
st.markdown("#### 📦 Bulk Message Prediction")
uploaded = st.file_uploader("Upload CSV (single column 'Msg' or with header) or TXT (one message per line)", type=["csv", "txt"])
has_header = st.checkbox("CSV contains header row?", value=False)

# placeholder for the dataframe so it can be updated in-place
place = st.empty()
df = None

if uploaded:
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded) if has_header else pd.read_csv(uploaded, header=None, names=["Msg"])
        else:
            lines = uploaded.getvalue().decode("utf-8", errors="ignore").splitlines()
            df = pd.DataFrame({"Msg": [x.strip() for x in lines if x.strip()]})
        place.dataframe(df)  # show initial
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        df = None

if df is not None and st.button("Predict Bulk", key="predict_bulk"):
    try:
        preds = interpret_preds(model.predict(df["Msg"].astype(str)))
        df["Prediction"] = preds
        place.dataframe(df)  # update in same place
        st.success("Bulk Prediction Completed!")
        # simple summary metrics
        counts = df["Prediction"].value_counts().to_dict()
        st.metric("Total messages", len(df))
        st.metric("Spam", counts.get("Spam", 0))
        st.metric("Ham", counts.get("Ham", 0))
        # download
        st.download_button(
            label="⬇ Download Predictions (CSV)",
            data=to_bytes_csv(df),
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Bulk prediction error: {e}")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Footer
# ---------------------------
st.write("")
st.markdown(f"""
<div style="text-align:center; color:#6b7280; margin-top:20px;">
    Built with 💙 using Streamlit · Designed & Developed by <b>{AUTHOR}</b>
    <br>
    <a href="https://linkedin.com/in/chandan-kumar-sah-752803387" target="_blank">LinkedIn</a> |
    <a href="https://github.com/ChankumarSah" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)
