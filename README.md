# Sistem Pakar Diagnosis Kerusakan Smartphone (Rule-Based)

UTS Praktikum INF313 - Kecerdasan Artifisial. Sistem pakar berbasis aturan (IF-THEN) dengan Python dan Streamlit
untuk membedakan kerusakan **hardware** dan **software** pada smartphone berdasarkan gejala, lengkap dengan
rekomendasi perbaikan.

## Fitur

| Komponen | Implementasi |
|---|---|
| Knowledge Base | `data/knowledge_base.json`: 26 gejala, 4 fakta antara, 17 kerusakan, 67 rule IF-THEN + CF + sumber |
| Inference Engine | Forward chaining (per level, urut spesifisitas lalu CF) dan backward chaining (goal-driven, bisa interaktif) |
| Working Memory | `WorkingMemory`: fakta user, fakta turunan, riwayat kontribusi rule, rule yang di-fire |
| User Interface | Streamlit: form gejala per kategori, validasi input, skor CF, grafik penalaran |
| Explanation | WHY (mengapa gejala dipakai/ditanyakan), HOW (rantai rule + CF kumulatif), jejak inferensi, graphviz |
| Certainty Factor | CF user (0.4-1.0), CF rule, `min(antecedent) x CF rule`, gabungan `CF1 + CF2 x (1 - CF1)` |
| Akuisisi Pengetahuan | Tambah/edit/hapus rule, validasi (ID, duplikat, elemen tak dikenal, CF, siklus), simpan ke JSON |
| Reporting & Logging | Riwayat konsultasi, statistik, rule paling sering, ekspor TXT/CSV/PDF |
| Pencarian & Filter | Cari gejala/kerusakan/rule, filter riwayat, database kerusakan lengkap |

## Instalasi dan Menjalankan

```bash
git clone <url-repo-kelompok>
cd smartphone_expert_system
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Menjalankan pengujian: `python -m pytest -q`

## Struktur Proyek

```
app.py                  # antarmuka Streamlit
src/
  certainty_factor.py   # rumus CF
  working_memory.py     # memori kerja
  knowledge_base.py     # Rule, KnowledgeBase, validasi, CRUD, pencarian
  inference_engine.py   # forward & backward chaining
  explanation.py        # WHY / HOW / trace / graphviz
  reporting.py          # riwayat, statistik, ekspor
data/knowledge_base.json
data/history.json       # log konsultasi (dibuat otomatis)
tests/test_system.py    # 23 tes, termasuk 8 kasus diagnosa
```

## Representasi Pengetahuan

```json
"R14": {"if": ["G01", "F01"], "then": "D01", "cf": 0.85, "description": "...", "source": "..."}
```

Semua kondisi di `if` harus terpenuhi (AND). Beberapa rule dengan `then` yang sama digabung dengan rumus CF paralel
(OR). `F01-F04` adalah fakta antara (contoh: F01 = indikasi kerusakan fisik akibat benturan) sehingga terbentuk
rantai dua tingkat: gejala -> fakta antara -> kerusakan.

## Cara Kerja Inferensi

**Forward chaining:** fakta awal dari pengguna masuk working memory. Rule dikelompokkan per level kedalaman
(dihitung dari ketergantungan antar rule). Tiap siklus, rule yang semua antecedent-nya memiliki CF >= 0.2 masuk
agenda, diurutkan berdasarkan jumlah kondisi lalu CF, lalu di-fire. Evidence = `min(CF antecedent) x CF rule`,
dan digabung ke CF fakta tujuan.

**Backward chaining:** dimulai dari hipotesis kerusakan. Sistem mencari rule yang menyimpulkannya, membuktikan
antecedent secara rekursif, dan menanyakan gejala yang belum diketahui lewat callback `ask(symptom, why)`.
Hasil CF-nya sama dengan forward chaining (diuji otomatis).

**Hardware vs software:** setiap kerusakan punya kategori. Sistem membandingkan CF tertinggi per kategori dan
menampilkan verdict Hardware, Software, atau Campuran bila selisih < 0.1.

## Contoh Output

```
Kesimpulan: Kerusakan Panel LCD/Display
Kepercayaan: 99%
Penjelasan:
1. R02: IF G02 (Layar retak atau pecah) THEN F01 (...) (CF: 0.95)
2. R11: IF G03 (...) THEN F04 (...) (CF: 0.70)
3. R17: IF G03 AND F01 THEN D02 (CF: 0.90)
4. R18: IF G02 AND G03 THEN D02 (CF: 0.90)
...
Rekomendasi: Hubungkan ke monitor eksternal ... Ganti panel LCD/OLED di teknisi.
```



