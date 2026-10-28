"""Explanation Facility: penjelasan WHY dan HOW beserta rantai penalaran."""
from __future__ import annotations

from .inference_engine import InferenceResult
from .knowledge_base import KnowledgeBase, Rule


class Explainer:
    """Menjelaskan mengapa gejala ditanyakan (WHY) dan bagaimana kesimpulan diperoleh (HOW)."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb

    def format_rule(self, rule: Rule) -> str:
        """Format rule: R14: IF G01 (...) AND F01 (...) THEN D01 (...) (CF: 0.85)."""
        ifs = " AND ".join(f"{a} ({self.kb.name(a)})" for a in rule.antecedents)
        return f"{rule.id}: IF {ifs} THEN {rule.consequent} ({self.kb.name(rule.consequent)}) (CF: {rule.cf:.2f})"

    # ------------------------------------------------------------------ WHY
    def why_symptom(self, symptom_id: str, limit: int = 3) -> str:
        """Alasan sebuah gejala ditanyakan: rule apa saja yang membutuhkannya."""
        rules = self.kb.rules_using(symptom_id)
        if not rules:
            return "Gejala ini belum dipakai oleh rule mana pun."
        targets: list[str] = []
        for r in rules:
            name = self.kb.name(r.consequent)
            if name not in targets:
                targets.append(name)
        sample = "; ".join(self.format_rule(r) for r in rules[:limit])
        extra = f" (+{len(rules) - limit} rule lain)" if len(rules) > limit else ""
        return (f"Gejala ini ditanyakan karena menjadi syarat pada {len(rules)} rule untuk menguji: "
                f"{', '.join(targets)}. Contoh: {sample}{extra}")

    # ------------------------------------------------------------------ HOW
    def how_steps(self, result: InferenceResult, fact_id: str) -> list[dict]:
        """Urutan rule (dari gejala ke kesimpulan) yang membentuk CF sebuah fakta."""
        steps: list[dict] = []
        seen: set[str] = set()

        def walk(fid: str) -> None:
            record = result.wm.facts.get(fid)
            if record is None or record.source != "inferred" or fid in seen:
                return
            seen.add(fid)
            for c in record.contributions:
                for a in c["antecedents"]:
                    walk(a)
            for c in record.contributions:
                rule = self.kb.rules[c["rule_id"]] if c["rule_id"] in self.kb.rules else None
                steps.append({
                    "rule_id": c["rule_id"],
                    "text": self.format_rule(rule) if rule else c["rule_id"],
                    "antecedents": c["antecedents"],
                    "evidence_cf": c["evidence_cf"],
                    "consequent": fid,
                    "cf_after": c["cf_after"],
                })

        walk(fact_id)
        return steps

    def how_text(self, result: InferenceResult, diagnosis_id: str, recommendation: bool = True) -> str:
        """Penjelasan HOW dalam bentuk teks siap tampil/ekspor."""
        cf = result.wm.get_cf(diagnosis_id)
        lines = [f"Kesimpulan: {self.kb.name(diagnosis_id)}",
                 f"Kepercayaan: {cf * 100:.0f}%", "Penjelasan:"]
        for i, s in enumerate(self.how_steps(result, diagnosis_id), 1):
            ants = ", ".join(f"{a}={v:.2f}" for a, v in s["antecedents"].items())
            lines.append(f"{i}. {s['text']}")
            lines.append(f"   fakta: {ants} -> evidence {s['evidence_cf']:.2f} -> CF kumulatif {s['cf_after']:.2f}")
        diag = self.kb.diagnoses.get(diagnosis_id)
        if recommendation and diag:
            lines.append("Rekomendasi: " + " ".join(diag["solutions"]))
        return "\n".join(lines)

    def how_dot(self, result: InferenceResult, diagnosis_id: str) -> str:
        """Pohon penalaran dalam format Graphviz DOT."""
        def esc(t: str) -> str:
            return t.replace('"', "'").replace("\\", "/")

        lines = ["digraph G {", "  rankdir=LR;", '  node [fontsize=10, shape=box, style="rounded,filled"];']
        nodes: set[str] = set()

        def node(fid: str) -> None:
            if fid in nodes:
                return
            nodes.add(fid)
            cf = result.wm.get_cf(fid)
            kind = self.kb.kind(fid)
            color = {"symptom": "#cfe8ff", "fact": "#fff3b0", "diagnosis": "#c8f2c8"}.get(kind or "", "#eeeeee")
            label = f"{fid}\\n{esc(self.kb.name(fid))}\\nCF {cf:.2f}"
            lines.append(f'  "{fid}" [label="{label}", fillcolor="{color}"];')

        for s in self.how_steps(result, diagnosis_id):
            rid = s["rule_id"]
            lines.append(f'  "{rid}_{s["consequent"]}" [label="{rid}\\nev {s["evidence_cf"]:.2f}", '
                         f'shape=ellipse, fillcolor="#f3f3f3"];')
            for a in s["antecedents"]:
                node(a)
                lines.append(f'  "{a}" -> "{rid}_{s["consequent"]}";')
            node(s["consequent"])
            lines.append(f'  "{rid}_{s["consequent"]}" -> "{s["consequent"]}";')
        lines.append("}")
        return "\n".join(lines)

    def trace_text(self, result: InferenceResult) -> str:
        """Jejak inferensi lengkap (reasoning path)."""
        return "\n".join(f"{t.step:>3}. {'  ' * t.depth}[{t.kind}] {t.text}" for t in result.trace)
