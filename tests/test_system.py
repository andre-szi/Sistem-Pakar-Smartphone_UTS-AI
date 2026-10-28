"""Pengujian sistem pakar: CF, knowledge base, inferensi, dan pelaporan."""
import shutil
from pathlib import Path

import pytest

from src.certainty_factor import combine_cf, combine_many, rule_evidence_cf
from src.explanation import Explainer
from src.inference_engine import InferenceEngine
from src.knowledge_base import KnowledgeBase, KnowledgeBaseError, Rule
from src.reporting import HistoryLogger, export_csv, export_pdf, export_txt

DATA = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


@pytest.fixture()
def kb(tmp_path):
    target = tmp_path / "kb.json"
    shutil.copyfile(DATA, target)
    return KnowledgeBase(target)


@pytest.fixture()
def engine(kb):
    return InferenceEngine(kb)


# ------------------------------------------------------------------ CF
def test_combine_cf_positive():
    assert combine_cf(0.8, 0.6) == pytest.approx(0.92)


def test_combine_cf_mixed_and_negative():
    assert combine_cf(0.8, -0.4) == pytest.approx(0.4 / 0.6, abs=1e-3)
    assert combine_cf(-0.5, -0.5) == pytest.approx(-0.75)


def test_combine_many_and_evidence():
    assert combine_many([0.5, 0.5]) == pytest.approx(0.75)
    assert rule_evidence_cf([0.8, 1.0], 0.9) == pytest.approx(0.72)


# ------------------------------------------------------- knowledge base
def test_kb_consistent(kb):
    assert len(kb.rules) >= 60
    assert kb.compute_levels()  # tidak ada siklus
    for rule in kb.rules.values():
        assert kb.validate_rule(rule, editing=True) == [], rule.id


def test_validation_errors(kb):
    assert kb.validate_rule(Rule("R01", ["G01"], "D01", 0.5))  # duplikat id
    assert kb.validate_rule(Rule("R99", [], "D01", 0.5))  # IF kosong
    assert kb.validate_rule(Rule("R99", ["GXX"], "D01", 0.5))  # gejala tak dikenal
    assert kb.validate_rule(Rule("R99", ["G01"], "G02", 0.5))  # THEN gejala
    assert kb.validate_rule(Rule("R99", ["G01"], "D01", 1.5))  # CF salah


def test_cycle_rejected(kb):
    kb.add_rule(Rule("R70", ["F01"], "F02", 0.5), persist=False)
    with pytest.raises(KnowledgeBaseError):
        kb.add_rule(Rule("R71", ["F02"], "F01", 0.5), persist=False)


def test_crud_persists(kb):
    kb.add_rule(Rule("R68", ["G12", "G26"], "D07", 0.6, "uji", "tes"))
    reloaded = KnowledgeBase(kb.path)
    assert "R68" in reloaded.rules
    kb.update_rule(Rule("R68", ["G12", "G26"], "D07", 0.7))
    assert KnowledgeBase(kb.path).rules["R68"].cf == 0.7
    kb.delete_rule("R68")
    assert "R68" not in KnowledgeBase(kb.path).rules


def test_search(kb):
    assert "G25" in kb.search_symptoms("jatuh")
    assert all(d["category"] == "Software" for d in kb.search_diagnoses(category="Software").values())
    assert kb.search_rules(consequent="D04")


# ------------------------------------------------- 8 test case diagnosa
CASES = [
    ("Layar retak + garis setelah jatuh", {"G25": 0.8, "G02": 1.0, "G03": 1.0}, "D02"),
    ("Baterai menggembung + panas", {"G08": 1.0, "G12": 0.8}, "D04"),
    ("Tidak bisa charge + port kotor", {"G09": 1.0, "G11": 0.8}, "D05"),
    ("Storage penuh + lag + force close", {"G15": 1.0, "G14": 0.8, "G13": 0.8}, "D08"),
    ("Bootloop setelah update", {"G16": 1.0, "G18": 0.8}, "D10"),
    ("Terkena air lalu mati total", {"G26": 1.0, "G17": 1.0}, "D07"),
    ("Sinyal hilang setelah update", {"G19": 1.0, "G18": 0.8}, "D17"),
    ("Baterai boros + persen turun drastis", {"G06": 0.8, "G07": 1.0}, "D03"),
]


@pytest.mark.parametrize("label,facts,expected", CASES, ids=[c[0] for c in CASES])
def test_forward_cases(engine, label, facts, expected):
    result = engine.forward_chaining(facts)
    assert result.top is not None, label
    assert result.top.diagnosis_id == expected, (label, [(c.diagnosis_id, c.cf) for c in result.conclusions])


def test_forward_cf_value(engine):
    # G08=1.0 -> R25: 0.9; R26 butuh G12 (tidak ada) -> hanya R25
    result = engine.forward_chaining({"G08": 1.0})
    assert result.top.diagnosis_id == "D04"
    assert result.top.cf == pytest.approx(0.9)
    # tambah G12=1.0: R25 (0.9) + R26 (0.85) -> 0.9 + 0.85*0.1 = 0.985
    both = engine.forward_chaining({"G08": 1.0, "G12": 1.0})
    assert both.wm.get_cf("D04") == pytest.approx(0.985)


def test_no_symptoms_no_conclusions(engine):
    assert engine.forward_chaining({}).conclusions == []


def test_category_summary(engine):
    hw = engine.forward_chaining({"G08": 1.0}).category_summary()
    assert hw["verdict"] == "Hardware"
    sw = engine.forward_chaining({"G15": 1.0, "G14": 1.0}).category_summary()
    assert sw["verdict"] == "Software"


def test_backward_equals_forward(engine):
    for _, facts, expected in CASES:
        fwd = engine.forward_chaining(facts)
        bwd = engine.backward_chaining(expected, facts)
        assert bwd.wm.get_cf(expected) == pytest.approx(fwd.wm.get_cf(expected)), expected


def test_backward_interactive_asks_with_why(engine):
    answers = {"G01": 1.0, "G25": 0.8}
    asked = []

    def ask(sid, why):
        asked.append((sid, why))
        return answers.get(sid)

    result = engine.backward_chaining("D01", {}, ask=ask)
    assert result.wm.get_cf("D01") > 0.5
    assert asked and all("rule" in why or "hipotesis" in why for _, why in asked)
    assert any(q["why"] for q in result.questions)


# ----------------------------------------------------------- explanation
def test_explanation(kb, engine):
    result = engine.forward_chaining(CASES[0][1])
    ex = Explainer(kb)
    text = ex.how_text(result, "D02")
    assert "R18" in text and "Kepercayaan" in text and "Rekomendasi" in text
    steps = ex.how_steps(result, "D02")
    assert steps[0]["rule_id"] in {"R01", "R02"}  # fakta antara dulu
    assert "digraph" in ex.how_dot(result, "D02")
    assert "G03" in ex.why_symptom("G03")
    assert ex.trace_text(result)


# ------------------------------------------------------------- reporting
def test_history_stats_and_exports(kb, engine, tmp_path):
    log = HistoryLogger(tmp_path / "h.json")
    facts = CASES[1][1]
    result = engine.forward_chaining(facts)
    rec = log.add("Budi", "Android X", facts, result)
    log.add("Sari", "", CASES[3][1], engine.forward_chaining(CASES[3][1]))
    assert log.stats()["total"] == 2
    assert "D04" == log.filter(name="budi")[0]["conclusions"][0]["id"]
    assert len(log.filter(diagnosis_id="D08")) == 1
    assert len(log.filter(symptom_id="G08")) == 1
    assert "Baterai Menggembung" in export_txt(rec, kb, result)
    assert export_csv(log.all()).count("\n") >= 3
    assert export_pdf(rec, kb, result)[:4] == b"%PDF"
    log.clear()
    assert log.all() == []
