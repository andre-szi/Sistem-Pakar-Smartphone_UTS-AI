"""Knowledge Base: representasi pengetahuan IF-THEN beserta validasi dan CRUD rule."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


class KnowledgeBaseError(Exception):
    """Kesalahan terkait knowledge base (file rusak, rule tidak valid)."""


@dataclass
class Rule:
    """Satu aturan IF (antecedent, semua harus terpenuhi) THEN (consequent) dengan CF."""

    id: str
    antecedents: list[str]
    consequent: str
    cf: float
    description: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        return {"if": list(self.antecedents), "then": self.consequent, "cf": self.cf,
                "description": self.description, "source": self.source}

    @classmethod
    def from_dict(cls, rule_id: str, data: dict) -> "Rule":
        return cls(rule_id, list(data["if"]), data["then"], float(data["cf"]),
                   data.get("description", ""), data.get("source", ""))


def _compute_levels(rules: dict[str, Rule], known_derived: set[str]) -> dict[str, int]:
    """Hitung level tiap rule (kedalaman rantai). Melempar error bila ada siklus."""
    by_consequent: dict[str, list[Rule]] = {}
    for rule in rules.values():
        by_consequent.setdefault(rule.consequent, []).append(rule)

    levels: dict[str, int] = {}
    visiting: set[str] = set()

    def level(rule: Rule) -> int:
        if rule.id in levels:
            return levels[rule.id]
        if rule.id in visiting:
            raise KnowledgeBaseError(f"Rule melingkar (sirkular) terdeteksi di {rule.id}")
        visiting.add(rule.id)
        deepest = 0
        for ant in rule.antecedents:
            if ant in known_derived:
                for sub in by_consequent.get(ant, []):
                    deepest = max(deepest, level(sub))
        visiting.discard(rule.id)
        levels[rule.id] = deepest + 1
        return levels[rule.id]

    for rule in rules.values():
        level(rule)
    return levels


class KnowledgeBase:
    """Memuat, menyimpan, mencari, dan memvalidasi pengetahuan sistem pakar."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self.meta: dict = {}
        self.symptoms: dict[str, dict] = {}
        self.facts: dict[str, dict] = {}
        self.diagnoses: dict[str, dict] = {}
        self.rules: dict[str, Rule] = {}
        self.load()

    # ------------------------------------------------------------------ I/O
    def load(self) -> None:
        """Muat knowledge base dari file JSON."""
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError as exc:
            raise KnowledgeBaseError(f"File knowledge base tidak ditemukan: {self.path}") from exc
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(f"File knowledge base bukan JSON valid: {exc}") from exc
        try:
            self.meta = data.get("meta", {})
            self.symptoms = data["symptoms"]
            self.facts = data["facts"]
            self.diagnoses = data["diagnoses"]
            self.rules = {rid: Rule.from_dict(rid, rd) for rid, rd in data["rules"].items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise KnowledgeBaseError(f"Struktur knowledge base tidak lengkap: {exc}") from exc

    def save(self) -> None:
        """Simpan ke file JSON secara atomik dan buat cadangan .bak."""
        data = {
            "meta": self.meta, "symptoms": self.symptoms, "facts": self.facts,
            "diagnoses": self.diagnoses,
            "rules": {rid: r.to_dict() for rid, r in sorted(self.rules.items())},
        }
        if self.path.exists():
            shutil.copyfile(self.path, self.path.with_suffix(".json.bak"))
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # --------------------------------------------------------------- lookup
    def kind(self, item_id: str) -> str | None:
        """Jenis elemen: symptom, fact, diagnosis, atau None."""
        if item_id in self.symptoms:
            return "symptom"
        if item_id in self.facts:
            return "fact"
        if item_id in self.diagnoses:
            return "diagnosis"
        return None

    def name(self, item_id: str) -> str:
        """Nama yang mudah dibaca untuk id gejala/fakta/diagnosis."""
        if item_id in self.symptoms:
            return self.symptoms[item_id]["text"]
        if item_id in self.facts:
            return self.facts[item_id]["text"]
        if item_id in self.diagnoses:
            return self.diagnoses[item_id]["name"]
        return item_id

    def derived_ids(self) -> set[str]:
        return set(self.facts) | set(self.diagnoses)

    def rules_for(self, consequent: str) -> list[Rule]:
        return [r for r in self.rules.values() if r.consequent == consequent]

    def rules_using(self, item_id: str) -> list[Rule]:
        return [r for r in self.rules.values() if item_id in r.antecedents]

    def compute_levels(self) -> dict[str, int]:
        return _compute_levels(self.rules, self.derived_ids())

    def symptom_categories(self) -> list[str]:
        seen: list[str] = []
        for s in self.symptoms.values():
            if s["category"] not in seen:
                seen.append(s["category"])
        return seen

    # ----------------------------------------------------------- validation
    def validate_rule(self, rule: Rule, editing: bool = False) -> list[str]:
        """Validasi rule. Mengembalikan daftar pesan error (kosong = valid)."""
        errors: list[str] = []
        rid = (rule.id or "").strip()
        if not rid:
            errors.append("ID rule tidak boleh kosong.")
        elif not rid.startswith("R") or not rid[1:].isdigit():
            errors.append("ID rule harus berformat R diikuti angka (contoh: R68).")
        elif not editing and rid in self.rules:
            errors.append(f"ID rule {rid} sudah dipakai.")
        elif editing and rid not in self.rules:
            errors.append(f"Rule {rid} tidak ditemukan untuk diedit.")

        if not rule.antecedents:
            errors.append("Bagian IF minimal berisi satu gejala/fakta.")
        if len(set(rule.antecedents)) != len(rule.antecedents):
            errors.append("Bagian IF mengandung elemen ganda.")
        for ant in rule.antecedents:
            if self.kind(ant) not in ("symptom", "fact"):
                errors.append(f"Antecedent '{ant}' tidak dikenal (bukan gejala/fakta).")
        if self.kind(rule.consequent) not in ("fact", "diagnosis"):
            errors.append("Bagian THEN harus berupa fakta antara atau diagnosis.")
        if rule.consequent in rule.antecedents:
            errors.append("Consequent tidak boleh sama dengan antecedent.")
        if not isinstance(rule.cf, (int, float)) or not 0 < rule.cf <= 1:
            errors.append("CF rule harus di antara 0 (tidak termasuk) hingga 1.")

        for other in self.rules.values():
            if other.id != rid and other.consequent == rule.consequent and \
                    set(other.antecedents) == set(rule.antecedents):
                errors.append(f"Rule duplikat dengan {other.id}.")
                break

        if not errors:
            trial = dict(self.rules)
            trial[rid] = rule
            try:
                _compute_levels(trial, self.derived_ids())
            except KnowledgeBaseError as exc:
                errors.append(str(exc))
        return errors

    # ----------------------------------------------------------------- CRUD
    def next_rule_id(self) -> str:
        numbers = [int(r[1:]) for r in self.rules if r[1:].isdigit()]
        return f"R{(max(numbers) + 1) if numbers else 1:02d}"

    def add_rule(self, rule: Rule, persist: bool = True) -> None:
        errors = self.validate_rule(rule)
        if errors:
            raise KnowledgeBaseError("; ".join(errors))
        self.rules[rule.id] = rule
        if persist:
            self.save()

    def update_rule(self, rule: Rule, persist: bool = True) -> None:
        errors = self.validate_rule(rule, editing=True)
        if errors:
            raise KnowledgeBaseError("; ".join(errors))
        self.rules[rule.id] = rule
        if persist:
            self.save()

    def delete_rule(self, rule_id: str, persist: bool = True) -> None:
        if rule_id not in self.rules:
            raise KnowledgeBaseError(f"Rule {rule_id} tidak ditemukan.")
        del self.rules[rule_id]
        if persist:
            self.save()

    # --------------------------------------------------------------- search
    def search_symptoms(self, query: str = "", category: str | None = None) -> dict[str, dict]:
        q = query.strip().lower()
        return {sid: s for sid, s in self.symptoms.items()
                if (not q or q in s["text"].lower() or q in sid.lower())
                and (not category or s["category"] == category)}

    def search_diagnoses(self, query: str = "", category: str | None = None,
                         severity: str | None = None) -> dict[str, dict]:
        q = query.strip().lower()
        result = {}
        for did, d in self.diagnoses.items():
            blob = f"{did} {d['name']} {d['description']}".lower()
            if q and q not in blob:
                continue
            if category and d["category"] != category:
                continue
            if severity and d["severity"] != severity:
                continue
            result[did] = d
        return result

    def search_rules(self, query: str = "", consequent: str | None = None) -> list[Rule]:
        q = query.strip().lower()
        found = []
        for r in self.rules.values():
            if consequent and r.consequent != consequent:
                continue
            blob = " ".join([r.id, r.description, self.name(r.consequent)]
                            + [self.name(a) for a in r.antecedents]).lower()
            if not q or q in blob:
                found.append(r)
        return sorted(found, key=lambda r: r.id)

    def diagnoses_for_symptoms(self, symptom_ids: list[str]) -> dict[str, list[str]]:
        """Diagnosis yang melibatkan gejala tertentu (langsung atau lewat fakta antara)."""
        reach: dict[str, set[str]] = {}
        for sid in symptom_ids:
            frontier = {sid}
            seen: set[str] = set()
            while frontier:
                item = frontier.pop()
                if item in seen:
                    continue
                seen.add(item)
                for rule in self.rules_using(item):
                    if self.kind(rule.consequent) == "diagnosis":
                        reach.setdefault(rule.consequent, set()).add(sid)
                    else:
                        frontier.add(rule.consequent)
        return {d: sorted(s) for d, s in reach.items()}
