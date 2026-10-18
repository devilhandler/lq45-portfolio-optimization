"""Menjalankan seluruh eksperimen dan menyimpan hasilnya ke folder results/.

Jalankan dari folder utama repo:  python -m src.eksperimen
"""
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
import pandas as pd

from .algoritma import ALGORITMA, brute_force, hill_climbing
from .masalah import (ROOT, UJI_MULAI, UJI_SELESAI, PortfolioProblem, baca_daftar_saham,
                      baca_harga)

HASIL = ROOT / "results"
BUDGETS = [300, 500, 1000, 2000]     # jatah evaluasi fungsi objektif per run
N_RUN = 30                           # tiap algoritma dijalankan 30 kali (seed 0-29)
TOL = 1e-9                           # dianggap "menemukan optimum" jika selisih < TOL


def eksperimen_utama(p, bf):
    """E1: bandingkan HC, SA, GA pada beberapa budget, masing-masing 30 run."""
    baris, kurva = [], {}
    for budget in BUDGETS:
        for nama, fungsi in ALGORITMA.items():
            riwayat = []
            for seed in range(N_RUN):
                r = fungsi(p, budget, np.random.default_rng(seed))
                baris.append(dict(algoritma=nama, budget=budget, run=seed, sharpe=r["sharpe"],
                                  optimum=abs(r["sharpe"] - bf["sharpe"]) < TOL,
                                  waktu_ms=r["waktu"] * 1000,
                                  saham=" ".join(p.info(r["state"])["saham"])))
                riwayat.append(r["riwayat"])
            if budget == max(BUDGETS):
                kurva[nama] = np.median(np.vstack(riwayat), axis=0)
    runs = pd.DataFrame(baris)
    runs.to_csv(HASIL / "hasil_semua_run.csv", index=False)
    pd.DataFrame(kurva).rename_axis("evaluasi").to_csv(HASIL / "konvergensi_median.csv")
    ringkas = (runs.groupby(["budget", "algoritma"])
               .agg(berhasil_persen=("optimum", lambda x: 100 * x.mean()),
                    sharpe_rata2=("sharpe", "mean"), sharpe_terburuk=("sharpe", "min"),
                    waktu_ms=("waktu_ms", "mean")).round(4).reset_index())
    ringkas.to_csv(HASIL / "ringkasan.csv", index=False)
    return runs, ringkas


def lanskap_hill_climbing(p, bf, n_pendakian=1000):
    """E2: dari 1000 titik awal acak, ke puncak mana hill climbing (tanpa restart) berakhir?"""
    rng = np.random.default_rng(2025)
    puncak = []
    for _ in range(n_pendakian):
        s = p.state_acak(rng)
        f = p.sharpe(s)
        langkah = 0
        while True:
            tet = p.semua_tetangga(s)
            nilai = [p.sharpe(t) for t in tet]
            j = int(np.argmax(nilai))
            if nilai[j] <= f:
                break
            s, f, langkah = tet[j], nilai[j], langkah + 1
        puncak.append(dict(sharpe=f, langkah=langkah, saham=" ".join(p.info(s)["saham"]),
                           global_=abs(f - bf["sharpe"]) < TOL))
    df = pd.DataFrame(puncak)
    df.to_csv(HASIL / "lanskap_hill_climbing.csv", index=False)
    return df


def uji_dunia_nyata(p, bf):
    """E3: beli portofolio terbaik pada 3 Feb 2025 lalu simpan sampai 15 Okt 2026."""
    harga = baca_harga().loc[UJI_MULAI:UJI_SELESAI]
    saham = p.info(bf["state"])["saham"]
    rel = harga / harga.iloc[0]
    nilai = pd.DataFrame({
        "Portofolio terbaik (5 saham)": rel[saham].mean(axis=1) * 100,      # modal 100, bobot sama
        "Semua 42 saham (bobot sama)": rel[p.tickers].mean(axis=1) * 100,
        "Indeks LQ45": rel["INDEKS_LQ45"] * 100,
    })
    nilai.round(3).to_csv(HASIL / "uji_dunia_nyata.csv", index_label="tanggal")
    per_saham = (rel[saham].iloc[-1] - 1).rename("return_periode_uji")
    return nilai, per_saham


def main():
    HASIL.mkdir(exist_ok=True)
    p = PortfolioProblem()
    bf = brute_force(p)
    info = p.info(bf["state"])
    nama = baca_daftar_saham()
    urut = np.argsort(bf["semua_nilai"])[::-1]
    print("Brute force:", info["saham"], f"Sharpe {info['sharpe']:.4f}",
          f"({bf['jumlah']:,} kombinasi, {bf['waktu']:.2f} detik)")

    runs, ringkas = eksperimen_utama(p, bf)
    print(ringkas.to_string(index=False))
    lanskap = lanskap_hill_climbing(p, bf)
    print(f"Lanskap: {lanskap.global_.mean():.0%} pendakian berakhir di puncak global, "
          f"{lanskap.saham.nunique()} puncak berbeda")
    nilai, per_saham = uji_dunia_nyata(p, bf)
    print("Uji dunia nyata (nilai akhir dari 100):", nilai.iloc[-1].round(1).to_dict())

    # pembanding "cara cepat": ambil 5 saham dengan Sharpe ratio sendiri-sendiri tertinggi
    sharpe_tunggal = (p.mu - p.rf) / np.sqrt(np.diag(p.cov))
    cepat = tuple(sorted(np.argsort(sharpe_tunggal)[::-1][:p.k]))
    info_cepat = p.info(cepat)
    info_cepat["peringkat"] = int((bf["semua_nilai"] > info_cepat["sharpe"] + 1e-9).sum()) + 1

    # angka penting untuk laporan & slide
    ringkasan = dict(
        cara_cepat=info_cepat,
        n_saham=p.n, k=p.k, n_kombinasi=int(bf["jumlah"]), waktu_brute_force=bf["waktu"],
        n_hari_latih=int(len(baca_harga().loc["2023-02-01":"2025-01-31"]) - 1),
        terbaik=info, peringkat_2=float(bf["semua_nilai"][urut[1]]),
        persen_kombinasi_positif=float((bf["semua_nilai"] > 0).mean() * 100),
        median_semua=float(np.median(bf["semua_nilai"])),
        nama_terbaik={s: nama.loc[s, "nama"] for s in info["saham"]},
        sektor_terbaik={s: nama.loc[s, "sektor"] for s in info["saham"]},
        lanskap=dict(persen_global=float(lanskap.global_.mean() * 100),
                     puncak_berbeda=int(lanskap.saham.nunique()),
                     langkah_rata2=float(lanskap.langkah.mean())),
        uji_akhir=nilai.iloc[-1].round(2).to_dict(),
        uji_per_saham=per_saham.round(4).to_dict(),
    )
    (HASIL / "ringkasan_angka.json").write_text(json.dumps(ringkasan, indent=1, ensure_ascii=False,
                                                           default=float))
    # 10 kombinasi terbaik hasil brute force (urutan sama dengan itertools.combinations)
    komb = list(combinations(range(p.n), p.k))
    rows = []
    for r, i in enumerate(urut[:10]):
        d = p.info(komb[i])
        rows.append(dict(peringkat=r + 1, saham=" ".join(d["saham"]), **{
            "return": round(d["return"], 4), "volatilitas": round(d["volatilitas"], 4),
            "sharpe": round(d["sharpe"], 4)}))
    pd.DataFrame(rows).to_csv(HASIL / "brute_force_10_terbaik.csv", index=False)
    np.save(HASIL / "brute_force_semua_sharpe.npy", bf["semua_nilai"].astype("float32"))


if __name__ == "__main__":
    main()
