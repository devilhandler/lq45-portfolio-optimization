"""Membuat semua grafik untuk laporan dan slide (PNG + PDF) di folder results/grafik/.

Jalankan setelah eksperimen:  python -m src.grafik
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .masalah import LATIH_MULAI, LATIH_SELESAI, ROOT, UJI_MULAI, baca_harga  # noqa: E402

HASIL = ROOT / "results"
GRAFIK = HASIL / "grafik"

WARNA = {"Hill Climbing": "#eda100", "Simulated Annealing": "#eb6834", "Genetic Algorithm": "#1baf7a"}
BIRU, UNGU, ABU = "#2a78d6", "#6250d6", "#898781"
INK, INK2 = "#0b0b0b", "#52514e"
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

plt.rcParams.update({
    "font.family": "Inter", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.titlepad": 8, "axes.labelsize": 9.5, "axes.labelcolor": INK2,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.7,
    "axes.axisbelow": True, "xtick.color": ABU, "ytick.color": ABU, "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2.0, "text.color": INK,
})


def koma(fmt):
    """Formatter angka dengan koma desimal (gaya Indonesia)."""
    return FuncFormatter(lambda v, _: format(v, fmt).replace(".", ",").replace("-", "−"))


def label_bulan(ax, index, setiap=6):
    """Label sumbu-x di hari bursa pertama tiap bulan Jan/Jul (atau tiap 3 bulan)."""
    awal_bulan = index.to_series().groupby([index.year, index.month]).first()
    pos = [d for d in awal_bulan if d.month % setiap == 1 and d != index[0]]
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{BULAN[d.month - 1]} {d.year}" for d in pos])


def simpan(fig, nama):
    GRAFIK.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRAFIK / f"{nama}.png", dpi=200, bbox_inches="tight")
    fig.savefig(GRAFIK / f"{nama}.pdf", bbox_inches="tight")
    plt.close(fig)


def g1_data():
    h = baca_harga()["INDEKS_LQ45"]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axvspan(pd.Timestamp(LATIH_MULAI), pd.Timestamp(LATIH_SELESAI), color="#cde2fb", alpha=0.6, lw=0)
    ax.axvspan(pd.Timestamp(UJI_MULAI), h.index[-1], color="#f1efe8", lw=0)
    ax.plot(h.index, h.values, color=INK2, lw=1.6)
    ax.text(pd.Timestamp("2024-02-01"), 121, "Data latih\n(untuk memilih saham)", ha="center",
            fontsize=9, color="#104281")
    ax.text(pd.Timestamp("2025-12-15"), 121, "Data uji\n(cek hasil)", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(55, 132)
    ax.set_ylabel("Indeks LQ45 (awal = 100)")
    ax.set_title("Indeks LQ45, Februari 2023 – Oktober 2026")
    label_bulan(ax, h.index)
    simpan(fig, "g1_data")


def g2_semua_kombinasi(r):
    v = np.load(HASIL / "brute_force_semua_sharpe.npy")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(v, bins=120, color="#86b6ef", edgecolor="white", linewidth=0.3)
    best = r["terbaik"]["sharpe"]
    ax.axvline(best, color=BIRU, lw=1.6)
    ax.annotate(f"Terbaik: Sharpe {best:.2f}\n{', '.join(r['terbaik']['saham'])}".replace(".", ","),
                xy=(best, ax.get_ylim()[1] * 0.55), xytext=(best - 0.08, ax.get_ylim()[1] * 0.8),
                ha="right", fontsize=9, color=BIRU,
                arrowprops=dict(arrowstyle="->", color=BIRU, lw=1))
    ax.set_xlabel("Sharpe ratio")
    ax.set_ylabel("Jumlah kombinasi")
    ax.xaxis.set_major_formatter(koma(".1f"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
    ax.set_title("Sharpe ratio seluruh 850.668 kombinasi 5 saham (hasil brute force)")
    simpan(fig, "g2_semua_kombinasi")


def g3_keberhasilan():
    s = pd.read_csv(HASIL / "ringkasan.csv")
    budgets = sorted(s.budget.unique())
    fig, ax = plt.subplots(figsize=(8, 3.4))
    lebar = 0.26
    for k, nama in enumerate(WARNA):
        y = [s[(s.budget == b) & (s.algoritma == nama)].berhasil_persen.iloc[0] for b in budgets]
        x = np.arange(len(budgets)) + (k - 1) * lebar
        bars = ax.bar(x, y, lebar - 0.03, color=WARNA[nama], label=nama)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 2, f"{yi:.0f}%", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(np.arange(len(budgets)))
    ax.set_xticklabels([f"{b:,}".replace(",", ".") + " evaluasi" for b in budgets])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Run yang menemukan\njawaban terbaik (%)")
    ax.set_title("Berapa kali dari 30 run algoritma menemukan portofolio terbaik?")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.grid(axis="x", visible=False)
    simpan(fig, "g3_keberhasilan")


def g4_konvergensi(r):
    k = pd.read_csv(HASIL / "konvergensi_median.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for nama in WARNA:
        ax.plot(np.arange(1, len(k) + 1), k[nama].values, color=WARNA[nama], label=nama)
    ax.axhline(r["terbaik"]["sharpe"], color=INK2, lw=1, ls="--")
    ax.text(1990, r["terbaik"]["sharpe"] + 0.04, "jawaban terbaik (brute force)", ha="right",
            fontsize=8.5, color=INK2)
    ax.set_xlim(0, 2000)
    ax.set_ylim(0.6, 1.95)
    ax.set_xlabel("Jumlah evaluasi fungsi objektif")
    ax.set_ylabel("Sharpe terbaik sejauh ini")
    ax.yaxis.set_major_formatter(koma(".1f"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
    ax.set_title("Kecepatan menemukan solusi (median dari 30 run)")
    ax.legend(loc="lower right")
    simpan(fig, "g4_konvergensi")


def g5_uji():
    u = pd.read_csv(HASIL / "uji_dunia_nyata.csv", index_col=0, parse_dates=True)
    warna = [BIRU, UNGU, ABU]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for c, w in zip(u.columns, warna):
        ax.plot(u.index, u[c], color=w, lw=1.8 if w == BIRU else 1.4, label=c)
        ax.text(u.index[-1] + pd.Timedelta(days=8), u[c].iloc[-1], f"{u[c].iloc[-1]:.0f}",
                va="center", fontsize=9, color=w)
    ax.axhline(100, color="#c3c2b7", lw=0.8)
    ax.set_ylabel("Nilai (modal awal = 100)")
    ax.set_title("Cek dunia nyata: dibeli 3 Feb 2025, disimpan sampai 15 Okt 2026")
    label_bulan(ax, u.index, setiap=3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8.5)
    simpan(fig, "g5_uji_dunia_nyata")


def main():
    r = json.loads((HASIL / "ringkasan_angka.json").read_text())
    g1_data()
    g2_semua_kombinasi(r)
    g3_keberhasilan()
    g4_konvergensi(r)
    g5_uji()
    print("grafik tersimpan di", GRAFIK)


if __name__ == "__main__":
    main()
