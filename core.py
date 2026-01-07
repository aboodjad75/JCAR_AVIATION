# -*- coding: utf-8 -*-
"""
core.py
-------
Core utilities for the JCAR RAG system.

Includes:
- init_openai_client
- list_available_domains
- load_domain_store
- embed_query
- top_k_similar (cosine similarity)
- diagnose_parts_distribution (diagnostic only; no LLM)
- answer_with_llm (STRICT prompt + Part scope lock)
- run_rag_query (STRICT Part filtering)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from config import EMBEDDING_MODEL, COMPLETION_MODEL

DATA_DIR = Path("data")


def _norm_part(x: Any) -> str:
    """Normalize part to a clean numeric string if possible (e.g., 'Part 0171' -> '171')."""
    if x is None:
        return ""
    s = str(x).strip().lower()
    if not s:
        return ""
    s = s.replace("jcar", "").replace("part", "").strip()

    # Try full int
    try:
        return str(int(s))
    except Exception:
        pass

    # Try leading digits
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            break
    if digits:
        try:
            return str(int(digits))
        except Exception:
            return digits

    return s


# =========================
# 1) OpenAI Client
# =========================

def init_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found.\n"
            "Create a .env file and add:\n"
            "OPENAI_API_KEY=sk-xxxx"
        )
    return OpenAI(api_key=api_key)


# =========================
# 2) Domains / Stores
# =========================

def list_available_domains() -> List[str]:
    domains = set()
    if not DATA_DIR.exists():
        return []
    for p in DATA_DIR.glob("*_embeddings.npy"):
        if p.name.endswith("_embeddings.npy"):
            domains.add(p.name.replace("_embeddings.npy", ""))
    return sorted(domains)


def load_domain_store(domain: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    emb_path = DATA_DIR / f"{domain}_embeddings.npy"
    meta_path = DATA_DIR / f"{domain}_metadata.json"

    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"No indexed files for domain: {domain}")

    embeddings = np.load(emb_path)

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    if len(metadata) != embeddings.shape[0]:
        raise RuntimeError("Mismatch: embeddings count != metadata count")

    return embeddings, metadata


# =========================
# 3) Embedding
# =========================

def embed_query(client: OpenAI, question: str) -> np.ndarray:
    res = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[question],
    )
    return np.array(res.data[0].embedding, dtype=np.float32)


# =========================
# 4) Similarity Search (cosine)
# =========================

def top_k_similar(
    query_vec: np.ndarray,
    embeddings: np.ndarray,
    k: int = 5,
) -> List[Tuple[int, float]]:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    E = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    sims = E @ q
    k = min(k, len(sims))

    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]

    return [(int(i), float(sims[i])) for i in idx]


# =========================
# 5) Diagnostic Only (no LLM)
# =========================

def diagnose_parts_distribution(
    client: OpenAI,
    question: str,
    selected_domain: str = "ALL",
    diag_k: int = 20,
) -> Dict[str, Any]:
    """
    Evidence-based diagnostic only:
    - retrieve top diag_k chunks
    - compute parts distribution in retrieved chunks
    - no LLM call
    """
    q_vec = embed_query(client, question)

    # Load data
    if selected_domain == "ALL":
        domains = list_available_domains()
        if not domains:
            return {"parts_ranked": [], "top_parts": [], "needs_multipart": False}

        all_emb = []
        all_meta = []
        for d in domains:
            e, m = load_domain_store(d)
            all_emb.append(e)
            all_meta.extend(m)

        if not all_emb:
            return {"parts_ranked": [], "top_parts": [], "needs_multipart": False}

        embeddings_all = np.vstack(all_emb)
        metadata_all = all_meta
    else:
        embeddings_all, metadata_all = load_domain_store(selected_domain)

    if embeddings_all.shape[0] == 0:
        return {"parts_ranked": [], "top_parts": [], "needs_multipart": False}

    results = top_k_similar(q_vec, embeddings_all, k=diag_k)

    parts = []
    for idx, _score in results:
        pn = _norm_part(metadata_all[idx].get("part_norm", None))
        if pn:
            parts.append(pn)

    if not parts:
        return {"parts_ranked": [], "top_parts": [], "needs_multipart": False}

    counts = Counter(parts)
    parts_ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_parts = [p for p, _c in parts_ranked[:3]]

    # Evidence threshold: if 2nd part appears enough times in topK
    needs_multipart = False
    if len(parts_ranked) >= 2:
        _, c2 = parts_ranked[1]
        needs_multipart = c2 >= max(2, int(0.35 * diag_k))

    return {
        "parts_ranked": parts_ranked,
        "top_parts": top_parts,
        "needs_multipart": needs_multipart,
    }


# =========================
# 6) LLM Answer (STRICT + Part Lock)
# =========================

def answer_with_llm(
    client: OpenAI,
    question: str,
    contexts: List[Dict[str, Any]],
    answer_mode: str = "SHORT",
    allowed_parts: Optional[List[str]] = None,
) -> str:
    # Build context blocks
    context_blocks = []
    parts_in_context = []

    allowed_norm_sorted: Optional[List[str]] = None
    if allowed_parts:
        allowed_norm_sorted = sorted({_norm_part(p) for p in allowed_parts if _norm_part(p)})

    for c in contexts:
        part_norm = _norm_part(c.get("part_norm", None)) or "?"
        domain = c.get("domain", "?")
        file = c.get("file", "?")
        text = (c.get("text", "") or "").strip()

        context_blocks.append(f"[Domain={domain} | Part={part_norm} | File={file}]\n{text}")

        if part_norm and part_norm not in {"?", ""}:
            parts_in_context.append(part_norm)

    full_context = "\n\n-----\n\n".join(context_blocks)

    main_part = "?"
    if parts_in_context:
        main_part = Counter(parts_in_context).most_common(1)[0][0]

    if answer_mode.upper() == "DETAILED":
        style_instruction = (
            "- Provide a concise structured answer with 3–7 bullet points.\n"
            "- Each bullet must be directly supported by the context.\n"
        )
    else:
        style_instruction = (
            "- Start with a direct answer (1–2 sentences).\n"
            "- Add up to 3 short bullets ONLY if needed.\n"
        )

    part_lock = ""
    if allowed_norm_sorted:
        part_lock = (
            "\nPART SCOPE LOCK (MANDATORY):\n"
            f"- The user selected Part(s): {allowed_norm_sorted}\n"
            "- You MUST ignore ANY chunk that is not in these Part(s), even if provided.\n"
            "- If a claim cannot be supported using ONLY these Part(s), say it is not explicitly specified.\n"
        )

    prompt = f"""
You are acting as a Civil Aviation Regulatory Inspector for the Jordan Civil Aviation Regulations (JCAR).

User question:
{question}

Regulatory context (EXCLUSIVE SOURCE OF TRUTH):
{full_context}

ABSOLUTE RULES (NON-NEGOTIABLE):
- You MUST use ONLY the regulatory context provided above.
- You MUST NOT use general knowledge, best practice, ICAO guidance, or other JCAR Parts not present in the context.
- You MUST NOT infer requirements. If it is not explicitly stated, you must say it is not explicitly specified.
- If the user's question mentions an entity, certificate, approval, or concept that is NOT explicitly defined or addressed in the provided context,
  you MUST write:
  "The provided regulatory context under JCAR Part {main_part} does not explicitly specify this."
  Then ONLY state what the provided context DOES explicitly say that is closest/relevant.

{part_lock}

STRICT REGULATORY WORDING:
- Use "shall" ONLY if the exact obligation is explicitly stated in the context.
- Use "must" ONLY when clearly restating an explicit obligation in the context.
- Use "should" ONLY for non-mandatory guidance explicitly indicated by the context.
- Do NOT use "shall/must" for anything not explicitly supported.

LANGUAGE:
- Answer in the SAME language as the user's question.

FORMAT:
{style_instruction}

QUESTION QUALITY CONTROL (ONLY IF NEEDED):
- If the question is underspecified AND the context is insufficient to answer precisely, add:
  "Inspector guidance – please clarify your question"
  and propose 2–4 concrete rephrased questions.
- Otherwise, omit guidance.

FINAL CHECK (MANDATORY):
- Verify every claim is traceable to the provided context (and to selected Part(s) if locked).
- If not traceable, remove it and replace with "not explicitly specified".
"""

    res = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return res.choices[0].message.content.strip()


# =========================
# 7) High-level RAG (STRICT Part filter)
# =========================

def run_rag_query(
    client: OpenAI,
    question: str,
    selected_domain: str = "ALL",
    answer_mode: str = "SHORT",
    top_k: int = 5,
    allowed_parts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    q_vec = embed_query(client, question)

    # Load data
    if selected_domain == "ALL":
        domains = list_available_domains()
        if not domains:
            raise RuntimeError("No data found in data/ folder.")

        all_emb = []
        all_meta = []
        for d in domains:
            e, m = load_domain_store(d)
            all_emb.append(e)
            all_meta.extend(m)

        if not all_emb:
            raise RuntimeError("No chunks available for retrieval.")

        embeddings_all = np.vstack(all_emb)
        metadata_all = all_meta
    else:
        embeddings_all, metadata_all = load_domain_store(selected_domain)

    if embeddings_all.shape[0] == 0:
        raise RuntimeError("No chunks available for retrieval.")

    # STRICT Part filter
    if allowed_parts:
        allowed_norm = set(_norm_part(p) for p in allowed_parts if _norm_part(p))

        keep_idx = []
        for i, m in enumerate(metadata_all):
            pn_raw = m.get("part_norm", None)

            # STRICT: drop missing/None/blank/unknown
            if pn_raw is None:
                continue

            pn = _norm_part(pn_raw)
            if not pn:
                continue

            if pn in allowed_norm:
                keep_idx.append(i)

        if not keep_idx:
            return {
                "answer": (
                    "[WARNING] No indexed JCAR text found for selected Part(s): "
                    f"{sorted(list(allowed_norm))}. Choose different Part(s) or remove Part filtering."
                ),
                "contexts": [],
                "debug_lines": [],
                "used_domain": selected_domain,
                "top_k": top_k,
            }

        embeddings_all = embeddings_all[keep_idx]
        metadata_all = [metadata_all[i] for i in keep_idx]

    # Retrieve
    results = top_k_similar(q_vec, embeddings_all, k=top_k)

    contexts: List[Dict[str, Any]] = []
    debug_lines: List[str] = []

    for idx, score in results:
        rec = dict(metadata_all[idx])
        rec["_score"] = float(score)
        contexts.append(rec)

        snippet = (rec.get("text", "") or "")[:250].replace("\n", " ")
        debug_lines.append(
            f"score={score:.4f} | part={_norm_part(rec.get('part_norm'))} | "
            f"domain={rec.get('domain')} | file={rec.get('file')} | "
            f"chunk={rec.get('chunk_index_in_file')} | {snippet}"
        )

    # LLM
    answer = answer_with_llm(
        client=client,
        question=question,
        contexts=contexts,
        answer_mode=answer_mode,
        allowed_parts=allowed_parts,
    )

    return {
        "answer": answer,
        "contexts": contexts,
        "debug_lines": debug_lines,
        "used_domain": selected_domain,
        "top_k": top_k,
    }
