"""
config.py
---------
ملف الإعدادات العامة لمشروع JCAR RAG.

- يحدد مجلد الملفات (DOCS_DIR)
- يربط كل Part بالدومين (PART_TO_DOMAIN)
- يحدد نوع الـ chunking لكل Part (PART_CHUNK_STRATEGY)
- يحدد أسماء الموديل المستخدم للـ embeddings
"""

from pathlib import Path

# 📂 مجلد الـ PDFs
DOCS_DIR = Path("docs")

# 🧠 موديل الإجابة (LLM) المستخدم لبناء الرد النهائي
COMPLETION_MODEL = "gpt-4.1-mini"


# 🧠 اسم موديل الـ Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"

# 1) تعريف الـ Domains وربطها بالـ parts
PART_TO_DOMAIN = {
    "1":   "DEFINITIONS",
    "5":   "UNITS",
    "11":  "RULEMAKING",
    "13":  "INVESTIGATION",
    "19":  "SAFETY",
    "25":  "SECURITY",

    "20":  "AIRWORTHINESS",
    "21":  "AIRWORTHINESS",
    "M":   "AIRWORTHINESS",
    "CS":  "AIRWORTHINESS",
    "47":  "AIRWORTHINESS",
    "MMEL": "AIRWORTHINESS",
    "145": "AIRWORTHINESS",


    "14":  "AERODROMES",
    "139": "AERODROMES",
    "77":  "AERODROMES",

    "FCL1": "LICENSING",
    "FCL2": "LICENSING",
    "FCL4": "LICENSING",
    "63":   "LICENSING",
    "65":   "LICENSING",
    "66":   "LICENSING",

    "MED":  "MEDICAL",
    "FCL3": "MEDICAL",


    "OPS1": "FLIGHT_OPS",
    "OPS3": "FLIGHT_OPS",
    "91":   "FLIGHT_OPS",
    "ARO":  "FLIGHT_OPS",
    "109":  "FLIGHT_OPS",
    "140":  "FLIGHT_OPS",
    "203":  "FLIGHT_OPS",

    "FSTD(A)": "TRAINING",
    "FSTD(H)": "TRAINING",
    "142": "TRAINING",
    "176": "TRAINING",
    "147": "TRAINING",


    "71":  "ANS",
    "73":  "ANS",
    "171": "ANS",
    "172": "ANS",
    "173": "ANS",
    "175": "ANS",
    "177": "ANS",

    "101": "UAS",
    "102": "UAS",

    "157": "ENVIRONMENT",
    "301": "ENVIRONMENT",

    "201": "ECON",
    "207": "ECON",
    "209": "ECON",
}

# 2) أنواع الـ chunking لكل Part
PART_CHUNK_STRATEGY = {
    # تعريفات واختصارات
    "1":        "GLOSSARY",      # Part 1 main definitions

    # مواد article-style
    "20":   "ARTICLE",
    "21":   "ARTICLE",
    "M":    "ARTICLE",
    "MMEL": "ARTICLE",
    "145":  "ARTICLE",
    "147":  "ARTICLE",
    "14":   "ARTICLE",
    "139":  "ARTICLE",
    "77":   "ARTICLE",
    "19":   "ARTICLE",
    "25":   "ARTICLE",
    "FCL1": "ARTICLE",
    "FCL2": "ARTICLE",
    "FCL3": "ARTICLE",
    "FCL4": "ARTICLE",
    "63":   "ARTICLE",
    "65":   "ARTICLE",
    "66":   "ARTICLE",
    "MED":  "ARTICLE",
    "OPS1": "ARTICLE",
    "OPS3": "ARTICLE",
    "91":   "ARTICLE",
    "ARO":  "ARTICLE",
    "109":  "ARTICLE",
    "140":  "ARTICLE",
    "203":  "ARTICLE",
    "142":  "ARTICLE",
    "171":  "ARTICLE",
    "172":  "ARTICLE",
    "173":  "ARTICLE",
    "175":  "ARTICLE",
    "176":  "ARTICLE",
    "177":  "ARTICLE",
    "71":   "ARTICLE",
    "73":   "ARTICLE",
    "101":  "ARTICLE",
    "102":  "ARTICLE",
    "157":  "ARTICLE",
    "301":  "ARTICLE",
    "201":  "ARTICLE",
    "207":  "ARTICLE",
    "209":  "ARTICLE",
    "47":  "ARTICLE",

    # أجزاء فيها جداول ثقيلة لو أردت تعامل خاص
    "5":    "TABLES_PER_BLOCK",   # Part 5: units → جدول لكل block مثلاً
}
