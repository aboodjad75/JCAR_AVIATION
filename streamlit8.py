#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
streamlit.py
-------------
User-controlled Domain + Part scope
Evidence-based warning (diagnostic Top-K on ALL)
Strict Part filtering via allowed_parts -> core.run_rag_query
Follow-up: yes/نعم/original text -> show original chunks without re-running RAG
Optional debug lines
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st

st.write("Mashallah 🚀 App is running...")

from core import (
    init_openai_client,
    list_available_domains,
    run_rag_query,
    diagnose_parts_distribution,
)

DATA_DIR = Path("data")


st.set_page_config(
    page_title="JCAR RAG – Inspector Chat",
    page_icon="✈️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .user-msg {
        background: #0f172a;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 10px 12px;
        margin: 6px 0;
        margin-left: auto;
        max-width: 85%;
        white-space: pre-wrap;
    }
    .ai-msg {
        background: #020617;
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 10px 12px;
        margin: 6px 0;
        margin-right: auto;
        max-width: 85%;
        white-space: pre-wrap;
    }
    .meta {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def has_arabic(text: str) -> bool:
    return any("\u0600" <= c <= "\u06FF" for c in text)


def render_message(role: str, text: str, meta: str = ""):
    rtl = has_arabic(text)
    direction = "rtl" if rtl else "ltr"
    align = "right" if rtl else "left"

    if role == "user":
        box = "user-msg"
        header = "👤 المستخدم"
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


ACK_WORDS = {
    "yes", "ok", "okay", "yep", "yeah",
    "نعم", "اه", "أه", "ايوه", "ايوا", "اها", "تمام",
    "show original", "original", "original text",
    "النص الاصلي", "النص الأصلي",
}


def is_ack_for_original(user_text: str, last_answer: Optional[str]) -> bool:
    if not last_answer:
        return False
    t = user_text.strip().lower()
    if not t:
        return False

    if t in ACK_WORDS:
        pass
    else:
        tokens = t.replace(",", " ").split()
        if not tokens:
            return False
        if tokens[0] not in ACK_WORDS and "original" not in t and "النص" not in t:
            return False

    last = last_answer.lower()
    markers = [
        "original jcar wording",
        "original jcar text",
        "original wording",
        "original text",
        "النص الأصلي",
        "النص الاصلي",
    ]
    return any(m in last for m in markers)


def build_original_text_from_contexts(contexts: List[Dict[str, Any]]) -> str:
    if not contexts:
        return "No underlying JCAR text is available for the previous answer."

    blocks = []
    for rec in contexts:
        part = rec.get("part_norm", "?")
        domain = rec.get("domain", "?")
        file = rec.get("file", "?")
        chunk_idx = rec.get("chunk_index_in_file", "?")
        text = (rec.get("text", "") or "").strip()
        header = f"[Domain={domain} | Part={part} | File={file} | Chunk={chunk_idx}]"
        blocks.append(f"{header}\n{text}")

    return "Here is the original JCAR text used in the previous answer:\n\n" + "\n\n-----\n\n".join(blocks)


def fmt_parts(parts: List[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return f"({parts[0]})"
    return "(" + ", ".join(parts) + ")"


# =========================
# Session state init
# =========================

if "client" not in st.session_state:
    st.session_state.client = init_openai_client()

if "domains" not in st.session_state:
    st.session_state.domains = list_available_domains()

if "parts" not in st.session_state:
    st.session_state.parts = list_available_parts_from_metadata()

if "chat" not in st.session_state:
    st.session_state.chat = []

if "last_diag_warning" not in st.session_state:
    st.session_state.last_diag_warning = ""

client = st.session_state.client
DOMAINS = st.session_state.domains
PARTS = st.session_state.parts


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.title("⚙️ Settings")

    selected_domain = st.selectbox(
        "Domain (user-controlled)",
        options=["ALL"] + DOMAINS,
        index=0,
        key="selected_domain",
        help="Domain=ALL قد يقلل الدقة بسبب تعدد Parts.",
    )

    answer_mode = st.radio(
        "Answer style",
        ["SHORT", "DETAILED"],
        format_func=lambda x: "Short (Inspector note)" if x == "SHORT" else "Detailed",
        key="answer_mode",
    )

    top_k = st.slider(
        "Top-K Retrieved Chunks (final answer)",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
        key="top_k",
    )

    st.divider()
    st.subheader("Part scope (user-controlled)")

    part_scope_mode = st.radio(
        "Part mode",
        ["No Part Filter", "Single-Part (Strict)", "Multi-Part (Select)"],
        index=0,
        key="part_scope_mode",
    )

    allowed_parts: Optional[List[str]] = None

    if part_scope_mode == "Single-Part (Strict)":
        if PARTS:
            p = st.selectbox("Select Part", PARTS, index=0, key="selected_part_single")
            allowed_parts = [p]
        else:
            st.info("No parts found in metadata.")
            allowed_parts = None

    elif part_scope_mode == "Multi-Part (Select)":
        if PARTS:
            ps = st.multiselect("Select Parts", PARTS, default=[], key="selected_parts_multi")
            allowed_parts = ps if ps else None
        else:
            st.info("No parts found in metadata.")
            allowed_parts = None

    st.divider()
    st.subheader("Evidence-based warning")

    diag_k = st.slider(
        "Diagnostic Top-K",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="diag_k",
        help="تشخيص فقط، لا يؤثر على الإجابة.",
    )

    show_debug = st.checkbox("Show debug lines", value=False, key="show_debug")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat.clear()
        st.session_state.last_diag_warning = ""
        st.rerun()


# =========================
# Main
# =========================

st.title("JCAR RAG – Regulatory Inspector")
st.caption("User-controlled scope (Domain/Part). Answers are strictly extracted from indexed JCAR text.")

if selected_domain == "ALL":
    st.warning("Domain = ALL قد يؤدي إلى نتائج من Parts متعددة. للحصول على دقة أعلى اختر Domain مناسب.")

if st.session_state.last_diag_warning:
    st.warning(st.session_state.last_diag_warning)

# show chat
for item in st.session_state.chat:
    meta = f"(Domain={item['domain']}, Mode={item['mode']})"
    ap = item.get("allowed_parts")
    pm = item.get("part_mode")
    if pm and ap:
        if pm == "Single-Part (Strict)":
            meta += f" | Part={ap[0]}"
        elif pm == "Multi-Part (Select)":
            meta += f" | Parts={','.join(ap)}"

    render_message("user", item["question"], meta=meta)
    render_message("assistant", item["answer"])

    if show_debug and item.get("debug_lines"):
        with st.expander("Debug lines"):
            st.code("\n".join(item["debug_lines"][:50]))

st.markdown("---")

with st.form("chat_form", clear_on_submit=True):
    question = st.text_area(
        "Ask your question:",
        height=120,
        placeholder="Example: As a CNS provider, what actions are required after a major system failure under JCAR Part 171?",
    )
    submit = st.form_submit_button("Send")

if submit:
    user_text = question.strip()
    if not user_text:
        st.warning("Please enter a question.")
        st.stop()

    last_item = st.session_state.chat[-1] if st.session_state.chat else None

    # Follow-up: show original text without re-running RAG
    if last_item and is_ack_for_original(user_text, last_item.get("answer", "")):
        orig_text = build_original_text_from_contexts(last_item.get("contexts", []))
        st.session_state.chat.append(
            {
                "question": user_text,
                "answer": orig_text,
                "domain": last_item["domain"],
                "mode": last_item["mode"],
                "top_k": last_item["top_k"],
                "contexts": last_item.get("contexts", []),
                "debug_lines": last_item.get("debug_lines", []),
                "part_mode": last_item.get("part_mode"),
                "allowed_parts": last_item.get("allowed_parts"),
            }
        )
        st.rerun()

    # Evidence-based diagnostic warning (always on ALL)
    try:
        diag = diagnose_parts_distribution(
            client=client,
            question=user_text,
            selected_domain="ALL",
            diag_k=diag_k,
        )
        if diag.get("needs_multipart") and len(diag.get("top_parts", [])) >= 2:
            st.session_state.last_diag_warning = (
                "Multi-Part overlap detected " + fmt_parts(diag["top_parts"]) + ". "
                "For higher precision, enable Multi-Part or choose the required Part(s)."
            )
        else:
            st.session_state.last_diag_warning = ""
    except Exception:
        st.session_state.last_diag_warning = ""

    # Run RAG
    with st.spinner("Analyzing JCAR context..."):
        result = run_rag_query(
            client=client,
            question=user_text,
            selected_domain=selected_domain,
            answer_mode=answer_mode,
            top_k=top_k,
            allowed_parts=allowed_parts,
        )

    st.session_state.chat.append(
        {
            "question": user_text,
            "answer": result["answer"],
            "domain": selected_domain,
            "mode": answer_mode,
            "top_k": top_k,
            "contexts": result.get("contexts", []),
            "debug_lines": result.get("debug_lines", []),
            "part_mode": part_scope_mode,
            "allowed_parts": allowed_parts,
        }
    )
    st.rerun()
