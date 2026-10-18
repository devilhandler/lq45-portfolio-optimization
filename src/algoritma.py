"""Tiga algoritma local search + brute force sebagai kunci jawaban.

Semua algoritma diberi jatah (budget) evaluasi fungsi objektif yang sama supaya adil.
Setiap algoritma mengembalikan dict berisi state terbaik, nilainya, dan riwayat
"nilai terbaik sejauh ini" setelah setiap evaluasi (untuk grafik konvergensi).
"""
from __future__ import annotations

import math
import time
from itertools import combinations

import numpy as np


class _Pencatat:
    """Membungkus fungsi objektif: menghitung evaluasi dan mencatat nilai terbaik."""

    def __init__(self, problem, budget):
        self.p, self.budget = problem, budget
        self.terbaik, self.nilai_terbaik, self.riwayat = None, -math.inf, []

    def habis(self) -> bool:
        return len(self.riwayat) >= self.budget

    def nilai(self, state) -> float:
        f = self.p.sharpe(state)
        if f > self.nilai_terbaik:
            self.terbaik, self.nilai_terbaik = state, f
        self.riwayat.append(self.nilai_terbaik)
        return f

    def hasil(self, nama, mulai, **lain):
        return dict(algoritma=nama, state=self.terbaik, sharpe=self.nilai_terbaik,
                    riwayat=np.array(self.riwayat), waktu=time.perf_counter() - mulai, **lain)


# ---------------------------------------------------------------- Hill Climbing
def hill_climbing(problem, budget, rng):
    """Steepest-ascent hill climbing dengan random restart.

    Dari state sekarang, periksa SEMUA tetangga lalu pindah ke tetangga terbaik.
    Kalau tidak ada tetangga yang lebih baik (puncak lokal), mulai lagi dari state acak.
    """
    mulai, c = time.perf_counter(), _Pencatat(problem, budget)
    restart = 0
    while not c.habis():
        sekarang = problem.state_acak(rng)
        f_sekarang = c.nilai(sekarang)
        while not c.habis():
            terbaik, f_terbaik = None, -math.inf
            for t in problem.semua_tetangga(sekarang):
                if c.habis():
                    break
                f = c.nilai(t)
                if f > f_terbaik:
                    terbaik, f_terbaik = t, f
            if f_terbaik > f_sekarang:
                sekarang, f_sekarang = terbaik, f_terbaik
            else:
                restart += 1          # terjebak di puncak lokal -> random restart
                break
    return c.hasil("Hill Climbing", mulai, restart=restart)


# ---------------------------------------------------------------- Simulated Annealing
def simulated_annealing(problem, budget, rng, t_awal=0.10, t_akhir=0.001):
    """Pilih satu tetangga acak. Kalau lebih baik, selalu diterima. Kalau lebih buruk,
    diterima dengan peluang e^(delta/T). Suhu T turun perlahan dari t_awal ke t_akhir."""
    mulai, c = time.perf_counter(), _Pencatat(problem, budget)
    alpha = (t_akhir / t_awal) ** (1 / budget)       # pendinginan geometrik
    sekarang = problem.state_acak(rng)
    f_sekarang = c.nilai(sekarang)
    T = t_awal
    while not c.habis():
        calon = problem.tetangga_acak(sekarang, rng)
        f_calon = c.nilai(calon)
        delta = f_calon - f_sekarang
        if delta > 0 or rng.random() < math.exp(delta / T):
            sekarang, f_sekarang = calon, f_calon
        T *= alpha
    return c.hasil("Simulated Annealing", mulai)


# ---------------------------------------------------------------- Genetic Algorithm
def genetic_algorithm(problem, budget, rng, ukuran_populasi=20, p_mutasi=0.3, elit=2):
    """Populasi berisi beberapa portofolio. Tiap generasi:
    1. Seleksi turnamen  : ambil 3 individu acak, pilih yang terbaik sebagai induk.
    2. Crossover         : saham yang dimiliki kedua induk diwariskan, sisanya diambil
                           acak dari saham induk lainnya (anak tetap berisi 5 saham).
    3. Mutasi            : dengan peluang p_mutasi, tukar 1 saham anak dengan saham acak.
    4. Elitisme          : 2 individu terbaik langsung masuk generasi berikutnya.
    """
    mulai, c = time.perf_counter(), _Pencatat(problem, budget)
    populasi = [problem.state_acak(rng) for _ in range(ukuran_populasi)]
    fitness = [c.nilai(s) for s in populasi]
    generasi = 0

    def turnamen():
        ikut = rng.choice(len(populasi), 3, replace=False)
        return populasi[max(ikut, key=lambda i: fitness[i])]

    while not c.habis():
        urut = np.argsort(fitness)[::-1]
        baru = [populasi[i] for i in urut[:elit]]
        f_baru = [fitness[i] for i in urut[:elit]]
        while len(baru) < ukuran_populasi and not c.habis():
            a, b = turnamen(), turnamen()
            sama = set(a) & set(b)
            sisa = list((set(a) | set(b)) - sama)
            anak = list(sama) + list(rng.choice(sisa, problem.k - len(sama), replace=False))
            anak = tuple(sorted(int(x) for x in anak))
            if rng.random() < p_mutasi:
                anak = problem.tetangga_acak(anak, rng)
            baru.append(anak)
            f_baru.append(c.nilai(anak))
        populasi, fitness = baru, f_baru
        generasi += 1
    return c.hasil("Genetic Algorithm", mulai, generasi=generasi)


# ---------------------------------------------------------------- Brute force
def brute_force(problem):
    """Hitung Sharpe ratio SEMUA kombinasi 5 dari 42 saham (850.668 kombinasi).
    Ini seperti uninformed search yang menelusuri seluruh ruang state: pasti menemukan
    jawaban terbaik, tetapi harus memeriksa semuanya."""
    mulai = time.perf_counter()
    kombinasi = np.array(list(combinations(range(problem.n), problem.k)))
    ret = problem.mu[kombinasi].mean(axis=1)
    var = np.zeros(len(kombinasi))
    for a in range(problem.k):                      # jumlahkan seluruh kovarians pasangan
        for b in range(problem.k):
            var += problem.cov[kombinasi[:, a], kombinasi[:, b]]
    var /= problem.k ** 2
    nilai = (ret - problem.rf) / np.sqrt(var)
    i = int(np.argmax(nilai))
    return dict(algoritma="Brute force", state=tuple(kombinasi[i]), sharpe=float(nilai[i]),
                semua_nilai=nilai, jumlah=len(kombinasi), waktu=time.perf_counter() - mulai)


ALGORITMA = {"Hill Climbing": hill_climbing, "Simulated Annealing": simulated_annealing,
             "Genetic Algorithm": genetic_algorithm}
