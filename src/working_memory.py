"""Working Memory: menyimpan fakta input pengguna dan hasil inferensi antara."""
from __future__ import annotations

from dataclasses import dataclass, field

from .certainty_factor import combine_cf


@dataclass
class FactRecord:
    """Satu fakta di working memory beserta riwayat kontribusi rule-nya."""

    fact_id: str
    cf: float
    source: str  # "user" atau "inferred"
    contributions: list[dict] = field(default_factory=list)


class WorkingMemory:
    """Penyimpanan fakta sementara yang diperbarui dinamis selama inferensi."""

    def __init__(self) -> None:
        self._facts: dict[str, FactRecord] = {}
        self._fired: list[str] = []

    @property
    def facts(self) -> dict[str, FactRecord]:
        return self._facts

    @property
    def fired_rules(self) -> list[str]:
        return list(self._fired)

    def set_user_fact(self, fact_id: str, cf: float) -> None:
        """Simpan fakta dari input pengguna (CF user)."""
        if not -1.0 <= cf <= 1.0:
            raise ValueError(f"CF {fact_id} di luar rentang [-1, 1]: {cf}")
        self._facts[fact_id] = FactRecord(fact_id, round(cf, 4), "user")

    def add_inferred(self, fact_id: str, evidence_cf: float, rule_id: str,
                     antecedents: dict[str, float]) -> tuple[float, float]:
        """Tambahkan/gabungkan fakta hasil inferensi. Mengembalikan (CF lama, CF baru)."""
        record = self._facts.get(fact_id)
        if record is None:
            record = FactRecord(fact_id, 0.0, "inferred")
            self._facts[fact_id] = record
        before = record.cf
        after = combine_cf(before, evidence_cf)
        record.cf = after
        record.contributions.append({
            "rule_id": rule_id,
            "antecedents": dict(antecedents),
            "evidence_cf": evidence_cf,
            "cf_before": before,
            "cf_after": after,
        })
        return before, after

    def mark_fired(self, rule_id: str) -> None:
        if rule_id not in self._fired:
            self._fired.append(rule_id)

    def has(self, fact_id: str) -> bool:
        return fact_id in self._facts

    def get_cf(self, fact_id: str) -> float:
        record = self._facts.get(fact_id)
        return record.cf if record else 0.0

    def snapshot(self) -> dict[str, float]:
        """Salinan sederhana {fact_id: cf}."""
        return {fid: rec.cf for fid, rec in self._facts.items()}
