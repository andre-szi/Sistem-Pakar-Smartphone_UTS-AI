"""Inference Engine: Forward Chaining dan Backward Chaining dengan Certainty Factor."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .certainty_factor import combine_cf, rule_evidence_cf
from .knowledge_base import KnowledgeBase, Rule
from .working_memory import WorkingMemory

# Callback untuk backward chaining interaktif: ask(symptom_id, alasan) -> CF atau None
AskCallback = Callable[[str, str], "float | None"]


@dataclass
class TraceStep:
    """Satu langkah jejak penalaran."""

    step: int
    kind: str  # AGENDA | FIRE | GOAL | SUBGOAL | ASK | SKIP | RESULT
    text: str
    rule_id: str | None = None
    cf_before: float | None = None
    cf_after: float | None = None
    depth: int = 0


@dataclass
class Conclusion:
    """Satu kesimpulan diagnosis beserta CF akhir."""

    diagnosis_id: str
    name: str
    category: str
    cf: float


@dataclass
class InferenceResult:
    """Hasil lengkap inferensi."""

    mode: str  # "forward" | "backward"
    wm: WorkingMemory
    trace: list[TraceStep]
    conclusions: list[Conclusion]
    goal: str | None = None
    questions: list[dict] = field(default_factory=list)

    @property
    def top(self) -> Conclusion | None:
        return self.conclusions[0] if self.conclusions else None

    @property
    def fired_rules(self) -> list[str]:
        return self.wm.fired_rules

    def category_summary(self, min_cf: float = 0.3) -> dict:
        """Bedakan hardware vs software berdasarkan CF diagnosis tertinggi per kategori."""
        best = {"Hardware": 0.0, "Software": 0.0}
        for c in self.conclusions:
            if c.category in best:
                best[c.category] = max(best[c.category], c.cf)
        hw, sw = best["Hardware"], best["Software"]
        if max(hw, sw) < min_cf:
            verdict = "Belum dapat disimpulkan"
        elif hw >= min_cf and sw >= min_cf and abs(hw - sw) < 0.1:
            verdict = "Campuran (Hardware dan Software)"
        else:
            verdict = "Hardware" if hw > sw else "Software"
        return {"Hardware": hw, "Software": sw, "verdict": verdict}


class InferenceEngine:
    """Mesin inferensi untuk basis pengetahuan berbasis rule."""

    def __init__(self, kb: KnowledgeBase, threshold: float = 0.2, min_conclusion_cf: float = 0.2):
        self.kb = kb
        self.threshold = threshold          # CF minimum agar sebuah fakta dianggap benar
        self.min_conclusion_cf = min_conclusion_cf

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _add(trace: list[TraceStep], kind: str, text: str, **kw) -> None:
        trace.append(TraceStep(step=len(trace) + 1, kind=kind, text=text, **kw))

    def _load_user_facts(self, wm: WorkingMemory, user_facts: dict[str, float]) -> None:
        for fid, cf in user_facts.items():
            if self.kb.kind(fid) == "symptom" and cf > 0:
                wm.set_user_fact(fid, cf)

    def _conclusions(self, wm: WorkingMemory, only: str | None = None) -> list[Conclusion]:
        result = []
        for did, d in self.kb.diagnoses.items():
            if only and did != only:
                continue
            cf = wm.get_cf(did)
            if cf >= self.min_conclusion_cf:
                result.append(Conclusion(did, d["name"], d["category"], cf))
        return sorted(result, key=lambda c: (-c.cf, c.diagnosis_id))

    def _eligible(self, rule: Rule, wm: WorkingMemory) -> bool:
        return all(wm.get_cf(a) >= self.threshold for a in rule.antecedents)

    # ----------------------------------------------------- forward chaining
    def forward_chaining(self, user_facts: dict[str, float]) -> InferenceResult:
        """Data-driven: mulai dari gejala, jalankan rule per level hingga tidak ada lagi."""
        wm = WorkingMemory()
        trace: list[TraceStep] = []
        self._load_user_facts(wm, user_facts)
        known = ", ".join(f"{fid}={cf:.1f}" for fid, cf in wm.snapshot().items()) or "-"
        self._add(trace, "AGENDA", f"Fakta awal di working memory: {known}")

        levels = self.kb.compute_levels()
        by_level: dict[int, list[Rule]] = defaultdict(list)
        for rule in self.kb.rules.values():
            by_level[levels[rule.id]].append(rule)

        cycle = 0
        for lvl in sorted(by_level):
            agenda = [r for r in by_level[lvl] if self._eligible(r, wm)]
            if not agenda:
                continue
            # Conflict resolution: rule paling spesifik, lalu CF tertinggi
            agenda.sort(key=lambda r: (-len(r.antecedents), -r.cf, r.id))
            cycle += 1
            self._add(trace, "AGENDA",
                      f"Siklus {cycle}: {len(agenda)} rule memenuhi syarat -> "
                      + ", ".join(r.id for r in agenda))
            for rule in agenda:
                ant = {a: wm.get_cf(a) for a in rule.antecedents}
                evidence = rule_evidence_cf(ant.values(), rule.cf)
                before, after = wm.add_inferred(rule.consequent, evidence, rule.id, ant)
                wm.mark_fired(rule.id)
                self._add(trace, "FIRE",
                          f"Fire {rule.id}: {self.kb.name(rule.consequent)} "
                          f"(evidence {evidence:.2f}; CF {before:.2f} -> {after:.2f})",
                          rule_id=rule.id, cf_before=before, cf_after=after)
        conclusions = self._conclusions(wm)
        top = conclusions[0] if conclusions else None
        self._add(trace, "RESULT",
                  f"Selesai. Kesimpulan teratas: {top.name} (CF {top.cf:.2f})" if top
                  else "Selesai. Tidak ada kesimpulan yang memenuhi ambang CF.")
        return InferenceResult("forward", wm, trace, conclusions)

    # ---------------------------------------------------- backward chaining
    def backward_chaining(self, goal: str, user_facts: dict[str, float],
                          ask: AskCallback | None = None) -> InferenceResult:
        """Goal-driven: buktikan hipotesis (diagnosis) mundur hingga ke gejala."""
        if self.kb.kind(goal) not in ("diagnosis", "fact"):
            raise ValueError(f"Goal '{goal}' bukan diagnosis/fakta antara.")
        wm = WorkingMemory()
        trace: list[TraceStep] = []
        questions: list[dict] = []
        self._load_user_facts(wm, user_facts)
        asked: set[str] = set(user_facts)
        proven: set[str] = set()
        self._add(trace, "GOAL", f"Hipotesis yang dibuktikan: {goal} - {self.kb.name(goal)}")

        def prove(fid: str, why: str, depth: int) -> float:
            kind = self.kb.kind(fid)
            if kind == "symptom":
                if fid not in asked:
                    asked.add(fid)
                    answer = ask(fid, why) if ask else None
                    cf = float(answer) if answer else 0.0
                    if cf > 0:
                        wm.set_user_fact(fid, cf)
                    questions.append({"symptom": fid, "why": why, "cf": cf})
                    self._add(trace, "ASK", f"Tanya {fid} ({self.kb.name(fid)}) karena {why} -> CF {cf:.1f}",
                              depth=depth)
                return wm.get_cf(fid)
            if fid in proven:
                return wm.get_cf(fid)
            proven.add(fid)
            for rule in self.kb.rules_for(fid):
                self._add(trace, "SUBGOAL", f"Coba {rule.id} untuk membuktikan {fid}", rule_id=rule.id,
                          depth=depth)
                ant: dict[str, float] = {}
                ok = True
                for a in rule.antecedents:
                    cf = prove(a, f"dibutuhkan rule {rule.id} untuk membuktikan {fid} "
                                  f"({self.kb.name(fid)})", depth + 1)
                    ant[a] = cf
                    if cf < self.threshold:
                        ok = False
                        self._add(trace, "SKIP", f"{rule.id} gagal: {a} tidak terbukti (CF {cf:.2f})",
                                  rule_id=rule.id, depth=depth)
                        break
                if ok:
                    evidence = rule_evidence_cf(ant.values(), rule.cf)
                    before, after = wm.add_inferred(fid, evidence, rule.id, ant)
                    wm.mark_fired(rule.id)
                    self._add(trace, "FIRE", f"{rule.id} terbukti: {fid} CF {before:.2f} -> {after:.2f}",
                              rule_id=rule.id, cf_before=before, cf_after=after, depth=depth)
            return wm.get_cf(fid)

        final = prove(goal, "ini adalah hipotesis utama", 0)
        conclusions = self._conclusions(wm, only=goal) if self.kb.kind(goal) == "diagnosis" else []
        self._add(trace, "RESULT", f"Hipotesis {goal} dibuktikan dengan CF {final:.2f}")
        return InferenceResult("backward", wm, trace, conclusions, goal=goal, questions=questions)
