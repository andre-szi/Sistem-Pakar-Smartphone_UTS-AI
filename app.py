"""Antarmuka Streamlit: Sistem Pakar Diagnosis Kerusakan Smartphone (Rule-Based)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.certainty_factor import USER_CF_LEVELS, interpret_cf
from src.explanation import Explainer
from src.inference_engine import InferenceEngine
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError, Rule
from src.reporting import HistoryLogger, export_csv, export_pdf, export_txt

st.set_page_config(page_title="Sistem Pakar Smartphone", page_icon="📱", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #554c63;
    --muted: #8b8298;
    --accent: #a58bc9;
    --accent-dark: #79639d;
    --accent-soft: #f0eafa;
    --line: #e9e2f0;
    --surface: #fffefe;
}

html, body {
    font-family: 'Poppins', sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background-color: #fcf9ff;
    background-image:
        radial-gradient(ellipse at 8% 5%, rgba(234, 218, 248, .38), transparent 33%),
        radial-gradient(ellipse at 94% 25%, rgba(218, 239, 232, .45), transparent 30%),
        radial-gradient(#eae2f1 0.6px, transparent 0.6px);
    background-size: auto, auto, 22px 22px;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { padding-top: 2.1rem; max-width: 1280px; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f2eafb 0%, #f5edf7 52%, #eaf5f1 100%);
    border-right: 1px solid #e7ddef;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #82768f; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-testid="stRadio"] label,
[data-testid="stSidebar"] [data-testid="stRadio"] label p {
    color: #625770 !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    min-height: 42px;
    margin: 3px 0;
    padding: 9px 12px;
    border: 1px solid transparent !important;
    border-radius: 11px;
    background: transparent !important;
    transition: background .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    border-color: #dfd1ee !important;
    background: #fffaff !important;
    box-shadow: 0 3px 10px rgba(119, 92, 151, .08);
}
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
    color: var(--ink) !important;
    font-weight: 600;
}
input, textarea, [data-baseweb="select"] *, [data-baseweb="input"] * {
    color: var(--ink) !important;
}
input::placeholder, textarea::placeholder { color: #778780 !important; opacity: 1; }

.brand-lockup {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 0 0 1.6rem;
    padding: 0 0 1.35rem;
    border-bottom: 1px solid #e2d6eb;
}
.brand-mark {
    display: grid;
    width: 44px;
    height: 44px;
    place-items: center;
    border-radius: 14px;
    background: linear-gradient(145deg, #e5daf5, #d9eee7);
    color: #79639d;
    font-size: 23px;
}
.brand-name { color: #554c63; font: 800 15px 'Poppins', sans-serif; }
.brand-caption { margin-top: 3px; color: #8b8298 !important; font-size: 11px; }

h1, h2, h3, [data-testid="stMetricValue"] {
    font-family: 'Poppins', sans-serif !important;
    color: var(--ink) !important;
}
h1 { font-size: 2rem !important; font-weight: 800 !important; letter-spacing: 0 !important; }
h2 { font-size: 1.35rem !important; font-weight: 750 !important; }
h3 { font-size: 1.08rem !important; font-weight: 700 !important; }
[data-testid="stCaptionContainer"] { color: var(--muted); }

.consult-hero {
    position: relative;
    overflow: hidden;
    margin: 0 0 1.6rem;
    padding: 1.65rem 1.8rem;
    border: 1px solid #e7dcef;
    border-radius: 20px;
    background: linear-gradient(112deg, #f0e8fa 0%, #f9f4fb 43%, #e8f4ef 76%, #fff1e9 100%);
    box-shadow: 0 12px 34px rgba(119, 92, 151, .08);
}
.page-hero {
    position: relative;
    overflow: hidden;
    margin: 0 0 1.25rem;
    padding: 1.4rem 1.65rem;
    border: 1px solid #e7dcef;
    border-radius: 18px;
    background: linear-gradient(112deg, #f0e8fa 0%, #f9f4fb 55%, #eaf5f0 100%);
    box-shadow: 0 10px 30px rgba(119, 92, 151, .07);
}
.page-hero .hero-title { margin: 0; color: var(--ink); font: 800 1.55rem 'Poppins', sans-serif; }
.page-hero .hero-description { max-width: 760px; margin: 6px 0 0; color: #81778d; font-size: 13px; }
.consult-hero::after {
    position: absolute;
    top: -54px;
    right: -24px;
    width: 190px;
    height: 190px;
    border: 1px solid rgba(22, 124, 104, .14);
    border-radius: 50%;
    box-shadow: 0 0 0 24px rgba(22, 124, 104, .035), 0 0 0 49px rgba(22, 124, 104, .025);
    content: '';
}
.hero-eyebrow {
    margin-bottom: 8px;
    color: #866da8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}
.consult-hero h1 { margin: 0 0 7px; font-size: 1.8rem !important; }
.consult-hero p { max-width: 650px; margin: 0; color: #81778d; font-size: 14px; }
.stats-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin: 0 0 1.15rem;
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: #fffefe;
    box-shadow: 0 5px 18px rgba(119, 92, 151, .055);
}
.stats-strip > div { min-width: 0; padding: 12px 16px; }
.stats-strip > div + div { border-left: 1px solid var(--line); }
.stats-label { display: block; color: var(--muted); font-size: 11px; font-weight: 700; }
.stats-value { display: block; margin-top: 2px; color: var(--ink); font: 800 1.4rem 'Poppins', sans-serif; }
.step-heading {
    display: flex;
    align-items: center;
    gap: 11px;
    margin: 1.4rem 0 .8rem;
}
.step-number {
    display: grid;
    width: 30px;
    height: 30px;
    flex: 0 0 30px;
    place-items: center;
    border-radius: 9px;
    background: var(--accent-soft);
    color: var(--accent-dark);
    font-size: 12px;
    font-weight: 800;
}
.step-title { color: var(--ink); font: 750 1.05rem 'Poppins', sans-serif; }
.step-note { margin-left: auto; color: var(--muted); font-size: 12px; text-align: right; }
.sidebar-summary {
    margin-top: 1.2rem;
    padding: 12px 13px;
    border: 1px solid #e1d5eb;
    border-radius: 12px;
    background: rgba(255, 252, 255, .78);
    color: #81778d;
    font-size: 12px;
    line-height: 1.8;
}
.sidebar-summary strong { color: #625770; font-weight: 700; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--line);
    border-radius: 10px;
}
[data-testid="stExpander"] {
    border: 1px solid #e9e2f0;
    border-radius: 13px;
    background: #fffefe;
    box-shadow: 0 3px 12px rgba(119, 92, 151, .045);
}
[data-testid="stExpander"] details > summary {
    border-radius: 13px;
    background: #fffefe !important;
    color: var(--ink) !important;
    font-weight: 650;
    transition: background .15s ease;
}
[data-testid="stExpander"] details[open] > summary {
    border-bottom: 1px solid var(--line);
    border-radius: 13px 13px 0 0;
    background: #f7f1fb !important;
}
[data-testid="stExpander"] details > div { background: #fffefe; }
[data-testid="stTabs"] [role="tablist"] { gap: 8px; border-bottom-color: var(--line); }
[data-testid="stTabs"] button[role="tab"] { color: var(--muted); font-weight: 600; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color: var(--accent-dark); }
[data-testid="stProgressBar"] > div > div { background: var(--accent); }
[data-testid="stExpander"] summary:hover { background: #f7f1fb !important; color: var(--accent-dark); }
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: var(--ink) !important; font-weight: 650; }
[data-testid="stRadio"] [role="radiogroup"] > label {
    padding: 8px 12px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #fffefe;
    transition: border-color .15s ease, background .15s ease;
}
[data-testid="stRadio"] input[type="radio"],
[data-testid="stCheckbox"] input[type="checkbox"] { accent-color: var(--accent); }
[data-baseweb="select"] > div {
    border-color: #e3d9ed !important;
    border-radius: 8px !important;
    background-color: #fffefe !important;
    color: var(--ink) !important;
}
[data-testid="stMultiSelect"] [role="group"] {
    min-height: 42px;
    border: 1px solid #e3d9ed !important;
    border-radius: 8px !important;
    background: #fffefe !important;
    color: var(--ink) !important;
}
[data-testid="stMultiSelect"] [data-testid="stMultiSelectTagsContainer"] {
    background: transparent !important;
    color: var(--ink) !important;
}
[data-testid="stMultiSelect"] button {
    background: transparent !important;
    color: var(--muted) !important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] div { color: var(--ink) !important; }
[data-baseweb="tag"] {
    border-radius: 8px !important;
    background: #f0eafa !important;
    color: var(--accent-dark) !important;
}
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [role="option"] {
    background: #fffefe !important;
    color: var(--ink) !important;
}
[role="listbox"],
[role="option"] {
    background: #fffefe !important;
    color: var(--ink) !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: #f0eafa !important;
    color: var(--accent-dark) !important;
}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"] {
    min-height: 42px;
    border-color: #e3d9ed;
    border-radius: 8px;
    background-color: #fffefe;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(165, 139, 201, .16);
}

.stButton > button, .stDownloadButton > button {
    min-height: 42px;
    border-radius: 8px;
    border: 1px solid #e3d9ed;
    color: var(--accent-dark);
    font-weight: 700;
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    border-color: var(--accent);
    box-shadow: 0 5px 14px rgba(119, 92, 151, .12);
}
.stButton > button[kind="primary"] {
    border-color: var(--accent);
    background: var(--accent);
    color: white;
}
.stButton > button[kind="primary"]:hover { background: var(--accent-dark); color: white; }
[data-testid="stMetric"] {
    padding: 14px 16px;
    border: 1px solid var(--line);
    border-radius: 9px;
    background: var(--surface);
    box-shadow: 0 3px 12px rgba(119, 92, 151, .055);
}
[data-testid="stMetricLabel"] { color: var(--muted); }
[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fffefe;
    box-shadow: 0 3px 12px rgba(119, 92, 151, .045);
}
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: var(--ink); }
div[data-testid="stForm"],
[data-testid="stVerticalBlockBorderWrapper"] {
    box-shadow: 0 3px 12px rgba(119, 92, 151, .045);
}
button:focus-visible, input:focus-visible, textarea:focus-visible,
[tabindex="0"]:focus-visible {
    outline: 2px solid var(--accent) !important;
    outline-offset: 2px;
}
hr { border-color: var(--line); }

@media (max-width: 700px) {
    [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 2rem; }
    .consult-hero { padding: 1.3rem 1.2rem; }
    .consult-hero h1 { font-size: 1.45rem !important; }
    .page-hero { padding: 1.2rem; }
    .page-hero .hero-title { font-size: 1.3rem; }
    .stats-strip > div { padding: 10px 8px; }
    .stats-label { font-size: 9px; line-height: 1.35; }
    .stats-value { font-size: 1.2rem; }
    .step-heading { flex-wrap: wrap; }
    .step-note { width: 100%; margin-left: 41px; text-align: left; }
    [data-testid="stRadio"] [role="radiogroup"] { gap: 6px; }
    [data-testid="stRadio"] [role="radiogroup"] > label { padding: 7px 9px; }
    [data-testid="stExpander"] { min-width: 0; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ state
def get_kb() -> KnowledgeBase:
    if "kb" not in st.session_state:
        try:
            st.session_state.kb = KnowledgeBase()
        except KnowledgeBaseError as exc:
            st.error(f"Gagal memuat knowledge base: {exc}")
            st.stop()
    return st.session_state.kb


kb = get_kb()
engine = InferenceEngine(kb)
explainer = Explainer(kb)
history = HistoryLogger()
SEVERITY_ICON = {"Ringan": "🟢", "Sedang": "🟡", "Berat": "🟠", "Berbahaya": "🔴"}


def item_label(item_id: str) -> str:
    return f"{item_id} - {kb.name(item_id)}"


def page_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
<section class="page-hero">
    <div class="hero-eyebrow">{eyebrow}</div>
    <h1 class="hero-title">{title}</h1>
    <p class="hero-description">{description}</p>
</section>
""",
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------- konsultasi
def render_result(record: dict, result) -> None:
    """Tampilkan hasil diagnosa, tingkat kepercayaan, penjelasan, dan ekspor."""
    summary = result.category_summary()
    st.subheader("Hasil Diagnosa")
    c1, c2, c3 = st.columns(3)
    c1.metric("Jenis masalah", summary["verdict"])
    c2.metric("Skor Hardware", f"{summary['Hardware'] * 100:.0f}%")
    c3.metric("Skor Software", f"{summary['Software'] * 100:.0f}%")

    if not result.conclusions:
        st.warning("Belum ada kerusakan yang memenuhi ambang kepercayaan. Tambahkan gejala lain atau "
                   "konsultasikan ke teknisi.")
    for i, c in enumerate(result.conclusions[:5], 1):
        diag = kb.diagnoses[c.diagnosis_id]
        icon = SEVERITY_ICON.get(diag["severity"], "")
        st.markdown(f"**{i}. {c.name}** ({c.category}) {icon} {diag['severity']}")
        st.progress(min(c.cf, 1.0), text=f"Kepercayaan {c.cf * 100:.0f}% - {interpret_cf(c.cf)}")
        if diag["severity"] == "Berbahaya":
            st.error("PERINGATAN: kondisi ini berbahaya. Hentikan penggunaan dan segera ke service center.")
        with st.expander(f"Detail, solusi, dan penjelasan HOW - {c.name}", expanded=(i == 1)):
            st.write(diag["description"])
            st.markdown("**Rekomendasi penanganan**")
            for s in diag["solutions"]:
                st.markdown(f"- {s}")
            st.markdown("**Pencegahan**")
            for p in diag["prevention"]:
                st.markdown(f"- {p}")
            st.markdown("**HOW - bagaimana sistem sampai pada kesimpulan ini**")
            for n, step in enumerate(explainer.how_steps(result, c.diagnosis_id), 1):
                ants = ", ".join(f"{a}={v:.2f}" for a, v in step["antecedents"].items())
                st.markdown(f"{n}. `{step['text']}`  \n   fakta: {ants} → evidence "
                            f"{step['evidence_cf']:.2f} → CF kumulatif **{step['cf_after']:.2f}**")
            st.graphviz_chart(explainer.how_dot(result, c.diagnosis_id))

    with st.expander("WHY - mengapa gejala ini dipakai sistem"):
        for sid in record["facts"]:
            st.markdown(f"**{item_label(sid)}**  \n{explainer.why_symptom(sid)}")
    if result.questions:
        with st.expander("Pertanyaan backward chaining (WHY)"):
            for q in result.questions:
                st.markdown(f"- {item_label(q['symptom'])}: {q['why']} (CF {q['cf']:.1f})")
    with st.expander("Jejak inferensi (reasoning trace)"):
        st.code(explainer.trace_text(result), language="text")
    with st.expander("Isi working memory"):
        st.dataframe(pd.DataFrame(
            [{"id": f.fact_id, "nama": kb.name(f.fact_id), "sumber": f.source, "CF": f.cf}
             for f in result.wm.facts.values()]), hide_index=True, width="stretch")

    st.markdown("**Ekspor laporan**")
    d1, d2, d3 = st.columns(3)
    d1.download_button("TXT", export_txt(record, kb, result), f"konsultasi_{record['id']}.txt")
    d2.download_button("CSV", export_csv([record]), f"konsultasi_{record['id']}.csv")
    try:
        d3.download_button("PDF", export_pdf(record, kb, result), f"konsultasi_{record['id']}.pdf")
    except Exception as exc:  # fpdf2 belum terpasang atau gagal render
        d3.caption(f"PDF tidak tersedia: {exc}")


def page_consult() -> None:
    st.markdown(
        """
<section class="consult-hero">
    <div class="hero-eyebrow">Pusat Diagnostik · Smartphone</div>
    <h1>Konsultasi kerusakan</h1>
    <p>Ceritakan gejala yang muncul. Sistem akan menelusuri kemungkinan penyebab dan memberi langkah penanganan yang sesuai.</p>
</section>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="stats-strip">
    <div><span class="stats-label">GEJALA TERDAFTAR</span><strong class="stats-value">{len(kb.symptoms)}</strong></div>
    <div><span class="stats-label">JENIS KERUSAKAN</span><strong class="stats-value">{len(kb.diagnoses)}</strong></div>
    <div><span class="stats-label">ATURAN DIAGNOSIS</span><strong class="stats-value">{len(kb.rules)}</strong></div>
</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    user_name = c1.text_input("Nama pengguna (opsional)")
    device = c2.text_input("Merek/tipe HP (opsional)")
    mode = st.radio("Metode inferensi", ["Forward Chaining (dari gejala)",
                                         "Backward Chaining (uji dugaan kerusakan)"], horizontal=True)
    goal = None
    if mode.startswith("Backward"):
        goal = st.selectbox("Dugaan kerusakan yang diuji", list(kb.diagnoses),
                            format_func=lambda d: f"{d} - {kb.diagnoses[d]['name']}")

    st.markdown(
        '<div class="step-heading"><span class="step-number">01</span>'
        '<span class="step-title">Pilih gejala</span>'
        '<span class="step-note">Pilih semua yang sesuai</span></div>',
        unsafe_allow_html=True,
    )
    selected: list[str] = []
    for category_index, cat in enumerate(kb.symptom_categories()):
        options = list(kb.search_symptoms(category=cat))
        key = f"ms_{cat}"
        selected_in_category = st.session_state.get(key, [])
        selected_count = len(selected_in_category)
        expander_title = f"{cat} · {selected_count} dipilih" if selected_count else cat
        with st.expander(expander_title, expanded=(category_index == 0)):
            selected += st.multiselect("Gejala", options, key=key, label_visibility="collapsed",
                                       format_func=lambda s: kb.symptoms[s]["text"])

    st.markdown(
        '<div class="step-heading"><span class="step-number">02</span>'
        '<span class="step-title">Tingkat keyakinan</span>'
        '<span class="step-note">Isi setelah gejala dipilih</span></div>',
        unsafe_allow_html=True,
    )
    user_facts: dict[str, float] = {}
    if not selected:
        st.info("Belum ada gejala dipilih.")
    for sid in selected:
        label = st.select_slider(kb.symptoms[sid]["text"], options=list(USER_CF_LEVELS), value="Yakin",
                                 key=f"cf_{sid}", help=explainer.why_symptom(sid))
        user_facts[sid] = USER_CF_LEVELS[label]

    if st.button("🔍 Diagnosa", type="primary"):
        if not user_facts:
            st.error("Pilih minimal satu gejala sebelum diagnosa.")
        else:
            try:
                result = (engine.forward_chaining(user_facts) if goal is None
                          else engine.backward_chaining(goal, user_facts))
                record = history.add(user_name, device, user_facts, result)
                st.session_state.last = (record, result)
            except (ValueError, KnowledgeBaseError, OSError) as exc:
                st.error(f"Terjadi kesalahan saat inferensi: {exc}")

    if "last" in st.session_state:
        st.divider()
        render_result(*st.session_state.last)


# ------------------------------------------------------- basis pengetahuan
def page_kb() -> None:
    page_header("Pustaka Sistem", "Basis pengetahuan",
                "Telusuri gejala, diagnosis, dan aturan yang menjadi dasar setiap hasil konsultasi.")
    t1, t2, t3, t4 = st.tabs(["Gejala", "Kerusakan", "Rules", "Cari dari gejala"])
    with t1:
        q = st.text_input("Cari gejala", key="q_sym")
        cat = st.selectbox("Kategori", [None] + kb.symptom_categories(), key="cat_sym",
                           format_func=lambda c: c or "Semua")
        rows = [{"ID": k, "Gejala": v["text"], "Kategori": v["category"]}
                for k, v in kb.search_symptoms(q, cat).items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with t2:
        q = st.text_input("Cari kerusakan", key="q_dg")
        c1, c2 = st.columns(2)
        cat = c1.selectbox("Kategori", [None, "Hardware", "Software"], key="cat_dg",
                           format_func=lambda c: c or "Semua")
        sev = c2.selectbox("Tingkat keparahan", [None, "Ringan", "Sedang", "Berat", "Berbahaya"],
                           key="sev_dg", format_func=lambda c: c or "Semua")
        for did, d in kb.search_diagnoses(q, cat, sev).items():
            with st.expander(f"{did} - {d['name']} ({d['category']}, {d['severity']})"):
                st.write(d["description"])
                st.markdown("**Solusi:** " + " ".join(d["solutions"]))
                st.markdown("**Pencegahan:** " + " ".join(d["prevention"]))
    with t3:
        q = st.text_input("Cari rule", key="q_rule")
        cons = st.selectbox("Filter THEN", [None] + sorted(kb.derived_ids()), key="cons_rule",
                            format_func=lambda c: "Semua" if c is None else item_label(c))
        found = kb.search_rules(q, cons)
        st.caption(f"{len(found)} dari {len(kb.rules)} rule")
        st.dataframe(pd.DataFrame(
            [{"Rule": r.id, "IF": " AND ".join(r.antecedents), "THEN": r.consequent, "CF": r.cf,
              "Keterangan": r.description, "Sumber": r.source} for r in found]),
            hide_index=True, width="stretch")
    with t4:
        picks = st.multiselect("Pilih gejala", list(kb.symptoms),
                               format_func=lambda s: kb.symptoms[s]["text"])
        if picks:
            for did, syms in kb.diagnoses_for_symptoms(picks).items():
                st.markdown(f"- **{kb.diagnoses[did]['name']}** ← " + ", ".join(syms))


# ------------------------------------------------------------- akuisisi
def rule_form(prefix: str, existing: Rule | None) -> Rule | None:
    """Form rule; mengembalikan Rule bila disubmit, selain itu None."""
    ant_options = list(kb.symptoms) + list(kb.facts)
    cons_options = list(kb.facts) + list(kb.diagnoses)
    with st.form(f"{prefix}_{existing.id if existing else 'new'}"):
        rid = existing.id if existing else kb.next_rule_id()
        st.text_input("ID Rule", rid, disabled=True)
        ants = st.multiselect("IF (semua harus terpenuhi)", ant_options,
                              default=existing.antecedents if existing else [], format_func=item_label)
        cons = st.selectbox("THEN", cons_options, format_func=item_label,
                            index=cons_options.index(existing.consequent) if existing else 0)
        cf = st.slider("CF rule", 0.05, 1.0, existing.cf if existing else 0.7, 0.05)
        desc = st.text_input("Keterangan", existing.description if existing else "")
        src = st.text_input("Sumber pengetahuan (pakar/literatur)", existing.source if existing else "")
        if st.form_submit_button("Simpan"):
            return Rule(rid, ants, cons, cf, desc, src)
    return None


def page_acquisition() -> None:
    page_header("Pengelolaan Sistem", "Akuisisi pengetahuan",
                "Kelola aturan IF-THEN yang digunakan sistem untuk menghubungkan gejala dengan diagnosis.")
    t1, t2, t3 = st.tabs(["Tambah rule", "Edit rule", "Hapus rule"])
    with t1:
        rule = rule_form("add", None)
        if rule:
            try:
                kb.add_rule(rule)
                st.success(f"Rule {rule.id} ditambahkan dan disimpan ke file.")
            except KnowledgeBaseError as exc:
                st.error(f"Rule tidak valid: {exc}")
    with t2:
        rid = st.selectbox("Pilih rule", sorted(kb.rules), key="edit_pick")
        if rid:
            rule = rule_form("edit", kb.rules[rid])
            if rule:
                try:
                    kb.update_rule(rule)
                    st.success(f"Rule {rule.id} diperbarui.")
                except KnowledgeBaseError as exc:
                    st.error(f"Rule tidak valid: {exc}")
    with t3:
        rid = st.selectbox("Pilih rule", sorted(kb.rules), key="del_pick")
        if rid:
            st.code(explainer.format_rule(kb.rules[rid]), language="text")
            if st.checkbox("Saya yakin ingin menghapus rule ini", key=f"del_ok_{rid}") and \
                    st.button("Hapus", type="primary"):
                kb.delete_rule(rid)
                st.success(f"Rule {rid} dihapus.")
                st.rerun()


# --------------------------------------------------------------- riwayat
def page_history() -> None:
    page_header("Rekam Konsultasi", "Riwayat dan statistik",
                "Pantau hasil konsultasi, pola diagnosis, dan penggunaan aturan dari waktu ke waktu.")
    t1, t2 = st.tabs(["Riwayat konsultasi", "Statistik"])
    with t1:
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Nama pengguna")
        diag = c2.selectbox("Diagnosa", [None] + list(kb.diagnoses), format_func=lambda d: "Semua"
                            if d is None else f"{d} - {kb.diagnoses[d]['name']}")
        sym = c3.selectbox("Gejala", [None] + list(kb.symptoms), format_func=lambda s: "Semua"
                           if s is None else item_label(s))
        c4, c5 = st.columns(2)
        verdict = c4.selectbox("Jenis masalah", [None, "Hardware", "Software", "Campuran (Hardware dan Software)"],
                               format_func=lambda v: v or "Semua")
        min_cf = c5.slider("CF minimum", 0.0, 1.0, 0.0, 0.05)
        records = history.filter(name, diag, verdict, sym, min_cf=min_cf)
        st.caption(f"{len(records)} konsultasi")
        st.dataframe(pd.DataFrame(
            [{"ID": r["id"], "Waktu": r["timestamp"], "Pengguna": r["user_name"],
              "Diagnosa utama": r["conclusions"][0]["name"] if r["conclusions"] else "-",
              "CF": r["conclusions"][0]["cf"] if r["conclusions"] else 0, "Jenis": r["verdict"]}
             for r in reversed(records)]), hide_index=True, width="stretch")
        if records:
            st.download_button("Ekspor semua (CSV)", export_csv(records), "riwayat.csv")
            pick = st.selectbox("Lihat detail", [r["id"] for r in reversed(records)])
            rec = next(r for r in records if r["id"] == pick)
            st.text(export_txt(rec, kb))
    with t2:
        stats = history.stats()
        m1, m2 = st.columns(2)
        m1.metric("Total konsultasi", stats["total"])
        m2.metric("Rata-rata CF diagnosa utama", f"{stats['avg_top_cf'] * 100:.0f}%")
        if stats["total"]:
            st.markdown("**Diagnosa paling sering**")
            st.bar_chart(pd.Series(stats["top_diagnoses"]))
            st.markdown("**Rule paling sering dipakai**")
            st.bar_chart(pd.Series(dict(list(stats["rule_usage"].items())[:15])))
            st.markdown("**Hardware vs Software**")
            st.bar_chart(pd.Series(stats["verdicts"]))
            if st.button("Hapus seluruh riwayat"):
                history.clear()
                st.rerun()
        else:
            st.info("Belum ada data konsultasi.")


def page_about() -> None:
    page_header("Informasi", "Tentang sistem",
                "Ringkasan pendekatan, cakupan, dan batasan sistem pakar diagnosis smartphone.")
    st.markdown(
        f"""
**{kb.meta.get('name', 'Sistem Pakar')}** — INF313 Kecerdasan Artifisial (Rule-Based, Python).

- {len(kb.symptoms)} gejala, {len(kb.facts)} fakta antara, {len(kb.diagnoses)} jenis kerusakan, {len(kb.rules)} rule.
- Inferensi: Forward Chaining (data-driven) dan Backward Chaining (goal-driven) dengan Certainty Factor.
- Penjelasan WHY/HOW, jejak penalaran, akuisisi pengetahuan, riwayat, statistik, ekspor TXT/CSV/PDF.

> {kb.meta.get('note', '')}
"""
    )


PAGES = {"Konsultasi": page_consult, "Basis Pengetahuan": page_kb,
         "Akuisisi Pengetahuan": page_acquisition, "Riwayat & Statistik": page_history,
         "Tentang": page_about}
st.sidebar.markdown(
    """
<div class="brand-lockup">
    <div class="brand-mark">⌁</div>
    <div><div class="brand-name">PhoneCare</div><div class="brand-caption">Sistem pakar smartphone</div></div>
</div>
""",
    unsafe_allow_html=True,
)
choice = st.sidebar.radio("Menu", list(PAGES))
st.sidebar.markdown(
    f"""
<div class="sidebar-summary">
    <strong>Cakupan basis data</strong><br>
    {len(kb.symptoms)} gejala · {len(kb.diagnoses)} diagnosis<br>
    {len(kb.rules)} aturan inferensi
</div>
""",
    unsafe_allow_html=True,
)
PAGES[choice]()
