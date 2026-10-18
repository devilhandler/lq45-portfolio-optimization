"""Formulasi masalah: memilih 5 saham LQ45 dengan Sharpe ratio tertinggi.

Masalah pencarian (sesuai modul Local Search):
- State      : himpunan 5 saham berbeda, misalnya (BBCA, BRIS, JPFA, MAPA, UNTR).
               Setiap saham diberi bobot sama, yaitu 20% modal.
- Tetangga   : tukar 1 saham di dalam portofolio dengan 1 saham di luar portofolio.
               Satu state punya 5 x 37 = 185 tetangga.
- Fungsi objektif : Sharpe ratio portofolio (makin besar makin baik).
- Tidak ada goal test: pencarian berhenti ketika jatah evaluasi (budget) habis.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Pengaturan utama proyek
K = 5                        # jumlah saham dalam portofolio
RF = 0.0575                  # suku bunga bebas risiko (BI-Rate Februari 2025) per tahun
LATIH_MULAI, LATIH_SELESAI = "2023-02-01", "2025-01-31"   # data untuk memilih saham
UJI_MULAI, UJI_SELESAI = "2025-02-03", "2026-10-15"       # data untuk mengecek hasilnya


def baca_harga() -> pd.DataFrame:
    """Harga harian (sudah disesuaikan dividen & stock split), dinormalisasi mulai dari 100."""
    return pd.read_csv(DATA / "harga_lq45.csv", index_col=0, parse_dates=True)


def baca_daftar_saham() -> pd.DataFrame:
    return pd.read_csv(DATA / "daftar_saham.csv", index_col=0)


class PortfolioProblem:
    """Menyimpan data return dan menghitung nilai objektif sebuah state."""

    def __init__(self, harga: pd.DataFrame | None = None, k: int = K, rf: float = RF,
                 mulai: str = LATIH_MULAI, selesai: str = LATIH_SELESAI):
        harga = baca_harga() if harga is None else harga
        self.tickers = [c for c in harga.columns if c != "INDEKS_LQ45"]
        r = harga.loc[mulai:selesai, self.tickers].pct_change().dropna()
        self.mu = r.mean().values * 252          # rata-rata return per tahun
        self.cov = r.cov().values * 252          # kovarians return per tahun
        self.n, self.k, self.rf = len(self.tickers), k, rf
        self.evaluasi = 0                        # penghitung pemanggilan fungsi objektif

    # ------------------------------------------------------------ fungsi objektif
    def sharpe(self, state) -> float:
        """Sharpe ratio = (return tahunan - RF) / volatilitas tahunan, bobot sama 1/K."""
        self.evaluasi += 1
        idx = list(state)
        ret = self.mu[idx].mean()
        var = self.cov[np.ix_(idx, idx)].sum() / self.k ** 2
        return (ret - self.rf) / np.sqrt(var)

    def info(self, state) -> dict:
        """Ringkasan portofolio (tidak dihitung sebagai evaluasi)."""
        idx = sorted(state)
        ret = self.mu[idx].mean()
        vol = np.sqrt(self.cov[np.ix_(idx, idx)].sum() / self.k ** 2)
        return {"saham": [self.tickers[i] for i in idx], "return": ret, "volatilitas": vol,
                "sharpe": (ret - self.rf) / vol}

    # ------------------------------------------------------------ state & tetangga
    def state_acak(self, rng) -> tuple:
        return tuple(sorted(rng.choice(self.n, self.k, replace=False)))

    def semua_tetangga(self, state) -> list[tuple]:
        """Semua state yang bisa dicapai dengan menukar 1 saham (185 tetangga)."""
        luar = [j for j in range(self.n) if j not in state]
        hasil = []
        for i in state:
            for j in luar:
                hasil.append(tuple(sorted([s for s in state if s != i] + [j])))
        return hasil

    def tetangga_acak(self, state, rng) -> tuple:
        """Satu tetangga acak: buang 1 saham acak, masukkan 1 saham acak dari luar."""
        keluar = rng.choice(state)
        luar = [j for j in range(self.n) if j not in state]
        masuk = rng.choice(luar)
        return tuple(sorted([s for s in state if s != keluar] + [masuk]))
