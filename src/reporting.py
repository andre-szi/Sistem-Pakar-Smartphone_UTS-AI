"""Reporting & Logging: riwayat konsultasi, statistik, dan ekspor TXT/CSV/PDF."""
from __future__ import annotations

import csv
import io
import json
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path

from .explanation import Explainer
from .inference_engine import InferenceResult
from .knowledge_base import KnowledgeBase

DEFAULT_HISTORY = Path(__file__).resolve().parent.parent / "data" / "history.json"


class HistoryLogger:
    """Menyimpan dan menganalisis riwayat konsultasi (file JSON)."""

    def __init__(self, path: str | Path = DEFAULT_HISTORY) -> None:
        self.path = Path(path)

    def all(self) -> list[dict]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _write(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)

    def add(self, user_name: str, device: str, user_facts: dict[str, float],
            result: InferenceResult) -> dict:
        """Catat satu konsultasi dan kembalikan record-nya."""
        summary = result.category_summary()
        record = {
            "id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "user_name": user_name.strip() or "Anonim",
            "device": device.strip(),
            "mode": result.mode,
            "goal": result.goal,
            "facts": {k: v for k, v in user_facts.items() if v > 0},
            "conclusions": [{"id": c.diagnosis_id, "name": c.name, "category": c.category, "cf": c.cf}
                            for c in result.conclusions],
            "rules_fired": result.fired_rules,
            "verdict": summary["verdict"],
        }
        records = self.all()
        records.append(record)
        self._write(records)
        return record

    def clear(self) -> None:
        self._write([])

    def filter(self, name: str = "", diagnosis_id: str | None = None, category: str | None = None,
               symptom_id: str | None = None, date_from: str | None = None,
               date_to: str | None = None, min_cf: float = 0.0) -> list[dict]:
        """Filter riwayat. Tanggal berformat YYYY-MM-DD."""
        out = []
        for rec in self.all():
            if name and name.lower() not in rec["user_name"].lower():
                continue
            if symptom_id and symptom_id not in rec["facts"]:
                continue
            day = rec["timestamp"][:10]
            if date_from and day < date_from:
                continue
            if date_to and day > date_to:
                continue
            concl = rec["conclusions"]
            if diagnosis_id and not any(c["id"] == diagnosis_id and c["cf"] >= min_cf for c in concl):
                continue
            if category and rec.get("verdict") != category:
                continue
            if min_cf and not diagnosis_id and not (concl and concl[0]["cf"] >= min_cf):
                continue
            out.append(rec)
        return out

    def stats(self) -> dict:
        """Statistik penggunaan sistem dan rule yang sering dipakai."""
        records = self.all()
        top_diag = Counter(r["conclusions"][0]["name"] for r in records if r["conclusions"])
        rules = Counter(rid for r in records for rid in r["rules_fired"])
        verdicts = Counter(r.get("verdict", "-") for r in records)
        symptoms = Counter(sid for r in records for sid in r["facts"])
        tops = [r["conclusions"][0]["cf"] for r in records if r["conclusions"]]
        return {
            "total": len(records),
            "avg_top_cf": round(sum(tops) / len(tops), 3) if tops else 0.0,
            "top_diagnoses": dict(top_diag.most_common()),
            "rule_usage": dict(rules.most_common()),
            "verdicts": dict(verdicts),
            "symptom_freq": dict(symptoms.most_common()),
        }


# ----------------------------------------------------------------- export
def export_txt(record: dict, kb: KnowledgeBase, result: InferenceResult | None = None) -> str:
    """Laporan konsultasi dalam teks. Bila result ada, sertakan penjelasan HOW."""
    lines = ["LAPORAN KONSULTASI - SISTEM PAKAR KERUSAKAN SMARTPHONE", "=" * 56,
             f"ID        : {record['id']}", f"Waktu     : {record['timestamp']}",
             f"Pengguna  : {record['user_name']}", f"Perangkat : {record['device'] or '-'}",
             f"Metode    : {'Forward' if record['mode'] == 'forward' else 'Backward'} chaining",
             f"Jenis     : {record['verdict']}", "", "GEJALA YANG DILAPORKAN"]
    for sid, cf in record["facts"].items():
        lines.append(f"- {sid} {kb.name(sid)} (CF user {cf:.1f})")
    lines += ["", "HASIL DIAGNOSA"]
    if not record["conclusions"]:
        lines.append("Tidak ada kesimpulan yang memenuhi ambang kepercayaan.")
    explainer = Explainer(kb)
    for i, c in enumerate(record["conclusions"], 1):
        lines.append(f"{i}. {c['name']} [{c['category']}] - kepercayaan {c['cf'] * 100:.0f}%")
        diag = kb.diagnoses.get(c["id"])
        if diag:
            lines.append(f"   Solusi: {' '.join(diag['solutions'])}")
        if result is not None and i == 1:
            lines.append("")
            lines.append(explainer.how_text(result, c["id"], recommendation=False))
    lines += ["", f"Rule yang digunakan: {', '.join(record['rules_fired']) or '-'}"]
    return "\n".join(lines)


def export_csv(records: list[dict]) -> str:
    """Ekspor daftar konsultasi ke CSV."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "waktu", "pengguna", "perangkat", "metode", "jenis", "diagnosa_utama",
                     "cf_utama", "gejala", "rule_digunakan"])
    for r in records:
        top = r["conclusions"][0] if r["conclusions"] else {"name": "-", "cf": 0}
        writer.writerow([r["id"], r["timestamp"], r["user_name"], r["device"], r["mode"], r["verdict"],
                         top["name"], top["cf"], " ".join(r["facts"]), " ".join(r["rules_fired"])])
    return buf.getvalue()


def export_pdf(record: dict, kb: KnowledgeBase, result: InferenceResult | None = None) -> bytes:
    """Ekspor laporan konsultasi ke PDF (membutuhkan fpdf2)."""
    from fpdf import FPDF  # impor lazy agar modul lain tetap jalan tanpa fpdf2

    def clean(text: str) -> str:
        return text.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, clean("Laporan Konsultasi Sistem Pakar Smartphone"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for line in export_txt(record, kb, result).split("\n")[2:]:
        pdf.multi_cell(0, 5.5, clean(line) or " ", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
