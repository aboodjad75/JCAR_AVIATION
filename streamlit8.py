from __future__ import annotations
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import streamlit as st

# تعطيل الـ watchers لتجنب انهيار cloud
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# هذا يجب أن يكون أول Streamlit call
st.set_page_config(
    page_title="JCAR RAG – Inspector Chat",
    page_icon="✈️",
    layout="wide",
)

# CSS لتغيير لون bubbles وصندوق الجواب
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; max-width: 1100px; }

/* User bubble */
.user-msg{
    background:#f3f4f6 !important;
    border:1px solid #e5e7eb !important;
    border-radius:14px;
    padding:10px 12px;
    margin:6px 0;
    margin-left:auto;
    max-width:85%;
    white-space:pre-wrap;
    color:#111827 !important;
}

/* Assistant bubble */
.ai-msg{
    background:#ffffff !important;
    border:1px solid #e5e7eb !important;
    border-radius:14px;
    padding:10px 12px;
    margin:6px 0;
    margin-right:auto;
    max-width:85%;
    white-space:pre-wrap;
    color:#111827 !important;
}

.meta{
    font-size:0.75rem;
    opacity:0.75;
    margin-bottom:6px;
    color:#374151 !important;
}
</style>
""", unsafe_allow_html=True)

# مجلدات البيانات
DATA_DIR = Path("data")

# دوال مساعدة
def has_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)

def render_message(role: str, text: str, meta: str = ""):
    rtl = has_arabic(text)
    direction = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"

    if role == "user":
        box = "user-msg"
        header = "👤 User"
    else:
        box = "ai-msg"
        header = "🛡️ JCAR Inspector"

    html = f"""
    <div dir="{direction}" style="text-align:{align}">
        <div class="{box}">
            <div class="meta">{header} {meta}</div>
            {text}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def list_available_parts_from_metadata() -> List[str]:
    parts = set()
    if not DATA_DIR.exists():
        return []
    for p in DATA_DIR.glob("*_metadata.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            for rec in meta:
                pn = rec.get("part_norm", None)
                if pn is None:
                    continue
                s = str(pn).strip()
                if not s or s in {"None", "?", ""}:
                    continue
                num = ""
                for ch in s:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num:
                    parts.add(str(int(num)))
                else:
                    parts.add(s)
        except Exception:
            continue

    def sort_key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)

    return sorted(parts, key=sort_key)

# تهيئة session state
if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_diag_warning" not in st.session_state:
    st.session_state.last_diag_warning = ""

# sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    DOMAINS = []  # لو عندك domains من core.py بترجعهم لاحقًا
    selected_domain = st.selectbox("Domain", ["ALL"] + DOMAINS, index=0)
    answer_mode = st.radio("Answer style", ["SHORT","DETAILED"], index=0)
    top_k = st.slider("Top-K", 3, 15, 5)
    diag_k = st.slider("Diagnostic Top-K", 10, 40, 20, step=5)
    show_debug = st.checkbox("Show debug lines", False)

    if st.button("🧹 Clear Chat"):
        st.session_state.chat.clear()
        st.session_state.last_diag_warning = ""
        st.rerun()

# عرض chat السابق
if st.session_state.last_diag_warning:
    st.warning(st.session_state.last_diag_warning)

st.title("JCAR RAG – Regulatory Inspector")
st.caption("Answers are extracted from indexed JCAR text only.")

for item in st.session_state.chat:
    meta = f"(Domain={item.get('domain')}, Mode={item.get('mode')})"
    render_message("user", item["question"], meta=meta)
    render_message("assistant", item["answer"])

    if show_debug and item.get("debug_lines"):
        with st.expander("Debug lines"):
            st.code("\n".join(item["debug_lines"][:50]))

st.markdown("---")

# form للسؤال الجديد
with st.form("chat_form", clear_on_submit=True):
    question = st.text_area(
        "Ask your question:",
        height=120,
        placeholder="Example: What are the safety obligations for fuel spillage under JCAR?",
    )
    submit = st.form_submit_button("Send")

if submit:
    user_text = question.strip()
    if not user_text:
        st.warning("Please enter a question.")
        st.stop()

    # خزّن السؤال
    st.session_state.chat.append({
        "question": user_text,
        "answer": "System is running, waiting for RAG integration...",  # مؤقت
        "domain": "ALL",
        "mode": answer_mode,
        "top_k": top_k,
        "contexts": [],
        "debug_lines": [],
        "part_mode": "",
        "allowed_parts": [],
    })

    # أعرض الجواب (مؤقت)
    safe = user_text.replace("\n","<br>")
    st.markdown(f"<div class='ai-msg'>{safe}</div>", unsafe_allow_html=True)
    st.rerun()
