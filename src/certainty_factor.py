"""Modul perhitungan Certainty Factor (CF).

Rentang CF yang dipakai: -1.0 hingga +1.0 (input pengguna: 0 hingga 1.0).
"""
from __future__ import annotations

from typing import Iterable

# Pilihan keyakinan pengguna terhadap sebuah gejala (CF User Input)
USER_CF_LEVELS: dict[str, float] = {
    "Ragu-ragu": 0.4,
    "Cukup yakin": 0.6,
    "Yakin": 0.8,
    "Sangat yakin": 1.0,
}


def clamp(cf: float) -> float:
    """Batasi nilai CF pada rentang [-1, 1]."""
    return max(-1.0, min(1.0, cf))


def cf_from_mb_md(mb: float, md: float) -> float:
    """CF(H,E) = MB(H,E) - MD(H,E)."""
    return round(clamp(mb - md), 4)


def combine_cf(cf1: float, cf2: float) -> float:
    """Gabungkan dua CF untuk hipotesis yang sama (multiple evidence).

    Kasus positif: CF = CF1 + CF2 * (1 - CF1). Kasus negatif dan campuran
    mengikuti rumus MYCIN.
    """
    if cf1 >= 0 and cf2 >= 0:
        result = cf1 + cf2 * (1 - cf1)
    elif cf1 < 0 and cf2 < 0:
        result = cf1 + cf2 * (1 + cf1)
    else:
        denom = 1 - min(abs(cf1), abs(cf2))
        result = (cf1 + cf2) / denom if denom else 0.0
    return round(clamp(result), 4)


def combine_many(cfs: Iterable[float]) -> float:
    """Gabungkan banyak CF secara berurutan."""
    total = 0.0
    for cf in cfs:
        total = combine_cf(total, cf)
    return total


def rule_evidence_cf(antecedent_cfs: Iterable[float], rule_cf: float) -> float:
    """CF evidence sebuah rule = min(CF antecedent) x CF rule."""
    values = list(antecedent_cfs)
    if not values:
        return 0.0
    return round(min(values) * rule_cf, 4)


def interpret_cf(cf: float) -> str:
    """Terjemahkan nilai CF menjadi label tingkat kepercayaan."""
    if cf >= 0.9:
        return "Sangat yakin"
    if cf >= 0.7:
        return "Yakin"
    if cf >= 0.5:
        return "Cukup yakin"
    if cf >= 0.3:
        return "Kurang yakin"
    return "Sangat lemah"
