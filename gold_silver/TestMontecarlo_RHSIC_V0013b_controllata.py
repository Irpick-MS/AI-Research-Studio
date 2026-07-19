#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RHSIC - Test Monte Carlo V0013b - ORO/ARGENTO controllata

Aggiunge rispetto alla V0012:
1) Suddivisione esplicita in ORO (14 sistemi originali) e ARGENTO (26 aggiuntivi).
2) Test separati per Oro e Argento.
3) Confronto dei risultati.

Nota metodologica:
I p-value valgono solo rispetto ai modelli nulli implementati.
Un p-value con 0 successi viene riportato come limite superiore p < 1/N.
"""

import os
import re
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import csv


# =============================================================================
# 1. DATI REALI - SUDDIVISIONE ORO / ARGENTO
# =============================================================================

# === ORO: 14 sistemi originali / campione base predefinito ===
ORO_DATA = [
    (14.0,   67.0,     'GRS 1915+105',          'stellar'),
    (6.3,    300.0,    'GRO J1655-40 (300)',    'stellar'),
    (6.3,    451.0,    'GRO J1655-40 (451)',    'stellar'),
    (9.1,    184.0,    'XTE J1550-564 (184)',   'stellar'),
    (9.1,    276.0,    'XTE J1550-564 (276)',   'stellar'),
    (7.8,    227.5,    'XTE J1859+226',         'stellar'),
    (428.0,  3.3,      'M82 X-1 (3.3)',         'imbh'),
    (428.0,  5.0,      'M82 X-1 (5.0)',         'imbh'),
    (5000.0, 0.30,     'NGC 1313 X-1 (0.30)',   'imbh'),
    (5000.0, 0.45,     'NGC 1313 X-1 (0.45)',   'imbh'),
    (1.7e6,  2.68e-4,  'RE J1034+396',          'agn'),
    (62.0,   264.0,    'GW150914',              'gw'),
    (20.8,   265.0,    'GW151226',              'gw'),
    (332.0,  62.0,     'GW190521',              'gw'),
]

# === ARGENTO: 26 sistemi aggiuntivi (incertezze maggiori, casi critici) ===
ARGENTO_DATA = [
    # Stellari aggiuntivi (10)
    (8.2,    250.0,    'XTE J1650-500',         'stellar'),
    (10.0,   160.0,    'H 1743-322 (160)',      'stellar'),
    (10.0,   240.0,    'H 1743-322 (240)',      'stellar'),
    (8.5,    200.0,    'XTE J1752-223',         'stellar'),
    (9.5,    210.0,    '4U 1630-47',            'stellar'),
    (6.0,    170.0,    'MAXI J1659-152',        'stellar'),
    (5.0,    140.0,    'GRO J0422+32',          'stellar'),
    (7.8,    150.0,    'XTE J1859+226 (150)',   'stellar'),
    (6.8,    130.0,    'XTE J1118+480',         'stellar'),
    (7.0,    110.0,    'XTE J1908+094',         'stellar'),
    # IMBH aggiuntivi (6)
    (2500.0, 0.13,     'NGC 5408 X-1 (0.13)',   'imbh'),
    (2500.0, 0.20,     'NGC 5408 X-1 (0.20)',   'imbh'),
    (5000.0, 0.20,     'Holmberg II X-1 (0.20)','imbh'),
    (5000.0, 0.30,     'Holmberg II X-1 (0.30)','imbh'),
    (5000.0, 0.10,     'NGC 1313 X-1 (0.10)',   'imbh'),
    (428.0,  6.5,      'M82 X-1 (6.5)',         'imbh'),
    # AGN aggiuntivi (4)
    (2.29e6, 2.60e-4,  '1H 0707-495',           'agn'),
    (1.0e6,  4.20e-4,  'MS 2254.9-3712',        'agn'),
    (1.8e6,  2.10e-4,  '2XMM J123103.2+110648', 'agn'),
    (1.2e6,  1.10e-4,  'Mrk 766',               'agn'),
    # GW aggiuntivi (6)
    (52.0,   230.0,    'GW170104',              'gw'),
    (18.5,   300.0,    'GW170608',              'gw'),
    (61.0,   250.0,    'GW170814',              'gw'),
    (28.0,   240.0,    'GW190412',              'gw'),
    (23.0,   280.0,    'GW190814',              'gw'),
    (7.0,    330.0,    'GW200115',              'gw'),
]

# === TOTALE ===
TOTAL_DATA = ORO_DATA + ARGENTO_DATA


# =============================================================================
# 2. PARAMETRI
# =============================================================================

M0_REF = 14.0
ZERO_LIMIT = 2500.0
N_ITER_MAIN = 100000
N_ITER_LOO = 30000
N_ITER_CLASS = 50000
THRESHOLD = 0.002
SEED = 42

TEST_TYPES = [
    "uniform_log_frequency",
    "permutation_frequency_mass"
]


# =============================================================================
# 3. LETTURA ZERI (uguale a V0012)
# =============================================================================

def find_zeros_file():
    possible_names = [
        'zeros_50000.txt',
        'zeros_5000.txt',
        'zeros_5000.csv',
        'zeros.csv',
        'zeros.txt',
        'zetazero.txt',
        'zeros1.txt'
    ]
    for name in possible_names:
        if os.path.isfile(name):
            return name
    return None


def load_zeros_any(filename, limit=2500.0):
    """
    Carica gli zeri non banali della zeta evitando di confondere gli indici
    di riga con i valori degli zeri.

    Formati supportati:
    1) una colonna:       14.134725...
    2) due colonne:       1  14.134725...
    3) formato mpf:       mpf('14.134725...')
    4) csv semplice:      1,14.134725...
    """
    zeros = []

    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    mpf_pattern = r"mpf\('([\d]+(?:\.[\d]+)?)'\)"

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Caso mpf('...')
        mpf_matches = re.findall(mpf_pattern, line)
        if mpf_matches:
            for val in mpf_matches:
                try:
                    t = float(val)
                    if 10.0 < t <= limit:
                        zeros.append(t)
                except ValueError:
                    pass
            continue

        # Caso CSV o testo con colonne: prende l'ultimo numero della riga.
        # In un file "indice valore", l'ultimo numero è il valore dello zero.
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", line)
        if not nums:
            continue

        try:
            t = float(nums[-1])
            if 10.0 < t <= limit:
                zeros.append(t)
        except ValueError:
            pass

    zeros = np.array(sorted(set(np.round(zeros, 12))), dtype=float)
    return zeros


# =============================================================================
# 4. FUNZIONI DI BASE (uguali a V0012)
# =============================================================================

def nearest_zero(phi, zeros):
    idx = np.searchsorted(zeros, phi)
    if idx == 0:
        return zeros[0]
    if idx >= len(zeros):
        return zeros[-1]
    left = zeros[idx - 1]
    right = zeros[idx]
    return left if abs(phi - left) <= abs(right - phi) else right


def relative_deviation(phi, zeros):
    z = nearest_zero(phi, zeros)
    return abs(phi - z) / z


def compute_stats(data, zeros, m0=M0_REF, threshold=THRESHOLD):
    phis = []
    znear = []
    devs = []
    for M, f, name, cls in data:
        phi = f * M / m0
        z = nearest_zero(phi, zeros)
        d = abs(phi - z) / z
        phis.append(phi)
        znear.append(z)
        devs.append(d)

    devs = np.array(devs)
    return {
        "phis": np.array(phis),
        "zeros": np.array(znear),
        "deviations": devs,
        "mean_dev": float(np.mean(devs)),
        "median_dev": float(np.median(devs)),
        "count_below": int(np.sum(devs < threshold)),
        "labels": [x[2] for x in data],
        "classes": [x[3] for x in data],
    }


# =============================================================================
# 5. MONTE CARLO
# =============================================================================

def monte_carlo(data, zeros, n_iter, test_type, seed, m0=M0_REF, threshold=THRESHOLD):
    rng = np.random.default_rng(seed)

    n = len(data)
    masses = np.array([x[0] for x in data], dtype=float)
    freqs = np.array([x[1] for x in data], dtype=float)

    log_freq_min = np.log10(np.min(freqs) * 0.5)
    log_freq_max = np.log10(np.max(freqs) * 2.0)

    mean_devs = np.zeros(n_iter)
    median_devs = np.zeros(n_iter)
    count_below = np.zeros(n_iter, dtype=int)

    for i in range(n_iter):
        if test_type == "uniform_log_frequency":
            f_rand = 10 ** rng.uniform(log_freq_min, log_freq_max, size=n)
        elif test_type == "permutation_frequency_mass":
            f_rand = rng.permutation(freqs)
        else:
            raise ValueError(f"Test type non riconosciuto: {test_type}")

        phis = f_rand * masses / m0
        devs = np.array([relative_deviation(phi, zeros) for phi in phis])

        mean_devs[i] = np.mean(devs)
        median_devs[i] = np.median(devs)
        count_below[i] = np.sum(devs < threshold)

    return {
        "mean_devs": mean_devs,
        "median_devs": median_devs,
        "count_below": count_below,
    }


def pvalue_less_equal(random_values, observed, n_iter):
    successes = int(np.sum(random_values <= observed))
    if successes == 0:
        return f"< {1/n_iter:.2e}", successes
    return f"{successes/n_iter:.6g}", successes


def pvalue_greater_equal(random_values, observed, n_iter):
    successes = int(np.sum(random_values >= observed))
    if successes == 0:
        return f"< {1/n_iter:.2e}", successes
    return f"{successes/n_iter:.6g}", successes


# =============================================================================
# 6. TEST PRINCIPALE SU UN DATASET
# =============================================================================

def run_main_tests(data, dataset_name, zeros):
    results = []
    obs = compute_stats(data, zeros)

    for k, test_type in enumerate(TEST_TYPES):
        print(f"\n[{dataset_name}] Test principale: {test_type}")
        mc = monte_carlo(data, zeros, N_ITER_MAIN, test_type, SEED + k)

        p_mean, s_mean = pvalue_less_equal(mc["mean_devs"], obs["mean_dev"], N_ITER_MAIN)
        p_med, s_med = pvalue_less_equal(mc["median_devs"], obs["median_dev"], N_ITER_MAIN)
        p_count, s_count = pvalue_greater_equal(mc["count_below"], obs["count_below"], N_ITER_MAIN)

        results.append({
            "dataset": dataset_name,
            "test": test_type,
            "n_systems": len(data),
            "observed_mean_dev_percent": obs["mean_dev"] * 100,
            "observed_median_dev_percent": obs["median_dev"] * 100,
            "observed_count_below": obs["count_below"],
            "p_mean": p_mean,
            "success_mean": s_mean,
            "p_median": p_med,
            "success_median": s_med,
            "p_count": p_count,
            "success_count": s_count,
        })

        plot_mc_distribution(
            mc["mean_devs"] * 100,
            obs["mean_dev"] * 100,
            f"[{dataset_name}] Monte Carlo {test_type}: media scarti",
            "Media scarti relativi (%)",
            f"rhsic_v0013_{dataset_name}_{test_type}_mean.png"
        )

        plot_mc_distribution(
            mc["count_below"],
            obs["count_below"],
            f"[{dataset_name}] Monte Carlo {test_type}: conteggio sotto soglia",
            f"Numero sistemi con delta < {THRESHOLD*100:.3f}%",
            f"rhsic_v0013_{dataset_name}_{test_type}_count.png"
        )

    return obs, results


# =============================================================================
# 7. LEAVE-ONE-OUT
# =============================================================================

def run_leave_one_out(data, dataset_name, zeros):
    rows = []
    print(f"\n[{dataset_name}] Leave-One-Out...")
    for i, removed in enumerate(data):
        reduced = data[:i] + data[i+1:]
        obs = compute_stats(reduced, zeros)

        row = {
            "dataset": dataset_name,
            "removed_system": removed[2],
            "removed_class": removed[3],
            "n_systems": len(reduced),
            "observed_mean_dev_percent": obs["mean_dev"] * 100,
            "observed_median_dev_percent": obs["median_dev"] * 100,
            "observed_count_below": obs["count_below"],
        }

        for k, test_type in enumerate(TEST_TYPES):
            mc = monte_carlo(reduced, zeros, N_ITER_LOO, test_type, SEED + 1000 + 10*i + k)

            p_mean, s_mean = pvalue_less_equal(mc["mean_devs"], obs["mean_dev"], N_ITER_LOO)
            p_med, s_med = pvalue_less_equal(mc["median_devs"], obs["median_dev"], N_ITER_LOO)
            p_count, s_count = pvalue_greater_equal(mc["count_below"], obs["count_below"], N_ITER_LOO)

            row[f"{test_type}_p_mean"] = p_mean
            row[f"{test_type}_success_mean"] = s_mean
            row[f"{test_type}_p_median"] = p_med
            row[f"{test_type}_success_median"] = s_med
            row[f"{test_type}_p_count"] = p_count
            row[f"{test_type}_success_count"] = s_count

        rows.append(row)

    return rows


# =============================================================================
# 8. TEST PER CLASSI
# =============================================================================

def run_class_tests(data, dataset_name, zeros):
    rows = []
    classes = sorted(set(x[3] for x in data))

    print(f"\n[{dataset_name}] Test per classi...")
    for cls in classes:
        subset = [x for x in data if x[3] == cls]
        if len(subset) < 2:
            continue

        obs = compute_stats(subset, zeros)
        row = {
            "dataset": dataset_name,
            "class": cls,
            "n_systems": len(subset),
            "observed_mean_dev_percent": obs["mean_dev"] * 100,
            "observed_median_dev_percent": obs["median_dev"] * 100,
            "observed_count_below": obs["count_below"],
        }

        for k, test_type in enumerate(TEST_TYPES):
            mc = monte_carlo(subset, zeros, N_ITER_CLASS, test_type, SEED + 2000 + 10*k + len(subset))

            p_mean, s_mean = pvalue_less_equal(mc["mean_devs"], obs["mean_dev"], N_ITER_CLASS)
            p_med, s_med = pvalue_less_equal(mc["median_devs"], obs["median_dev"], N_ITER_CLASS)
            p_count, s_count = pvalue_greater_equal(mc["count_below"], obs["count_below"], N_ITER_CLASS)

            row[f"{test_type}_p_mean"] = p_mean
            row[f"{test_type}_success_mean"] = s_mean
            row[f"{test_type}_p_median"] = p_med
            row[f"{test_type}_success_median"] = s_med
            row[f"{test_type}_p_count"] = p_count
            row[f"{test_type}_success_count"] = s_count

        rows.append(row)

    return rows


# =============================================================================
# 9. M0 SCAN
# =============================================================================

def run_m0_scan(data, dataset_name, zeros, m0_min=10.0, m0_max=18.0, step=0.25):
    rows = []
    m0_values = np.arange(m0_min, m0_max + 0.0001, step)

    for m0 in m0_values:
        obs = compute_stats(data, zeros, m0=m0)
        rows.append({
            "dataset": dataset_name,
            "M0": m0,
            "mean_dev_percent": obs["mean_dev"] * 100,
            "median_dev_percent": obs["median_dev"] * 100,
            "count_below": obs["count_below"],
        })

    return rows


# =============================================================================
# 10. OUTPUT
# =============================================================================

def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_individual_results(path, data, obs, dataset_name):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "system", "class", "M", "frequency", "Phi", "nearest_zero", "delta_percent"])
        for (M, freq, name, cls), phi, z, d in zip(data, obs["phis"], obs["zeros"], obs["deviations"]):
            writer.writerow([dataset_name, name, cls, M, freq, phi, z, d * 100])


def write_summary(path, obs_oro, results_oro, obs_argento, results_argento, loo_oro, loo_argento, class_oro, class_argento):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RHSIC - Monte Carlo V0013b\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"M0: {M0_REF}\n")
        f.write(f"Soglia: {THRESHOLD*100:.4f}%\n")
        f.write(f"Iterazioni main: {N_ITER_MAIN}\n")
        f.write(f"Iterazioni leave-one-out: {N_ITER_LOO}\n")
        f.write(f"Iterazioni classi: {N_ITER_CLASS}\n")
        f.write(f"Sistemi ORO: {len(ORO_DATA)}\n")
        f.write(f"Sistemi ARGENTO: {len(ARGENTO_DATA)}\n")
        f.write(f"Sistemi TOTALI: {len(TOTAL_DATA)}\n\n")

        f.write("-" * 80 + "\n")
        f.write("ORO - Statistiche osservate:\n")
        f.write(f"  Media scarti:   {obs_oro['mean_dev']*100:.6f}%\n")
        f.write(f"  Mediana scarti: {obs_oro['median_dev']*100:.6f}%\n")
        f.write(f"  Count sotto soglia: {obs_oro['count_below']} / {len(ORO_DATA)}\n")
        f.write("\n  p-value ORO:\n")
        for r in results_oro:
            f.write(f"\n    Test: {r['test']}\n")
            f.write(f"      P(media_random <= media_obs):     {r['p_mean']}  successi={r['success_mean']}\n")
            f.write(f"      P(mediana_random <= mediana_obs): {r['p_median']}  successi={r['success_median']}\n")
            f.write(f"      P(count_random >= count_obs):     {r['p_count']}  successi={r['success_count']}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("ARGENTO - Statistiche osservate:\n")
        f.write(f"  Media scarti:   {obs_argento['mean_dev']*100:.6f}%\n")
        f.write(f"  Mediana scarti: {obs_argento['median_dev']*100:.6f}%\n")
        f.write(f"  Count sotto soglia: {obs_argento['count_below']} / {len(ARGENTO_DATA)}\n")
        f.write("\n  p-value ARGENTO:\n")
        for r in results_argento:
            f.write(f"\n    Test: {r['test']}\n")
            f.write(f"      P(media_random <= media_obs):     {r['p_mean']}  successi={r['success_mean']}\n")
            f.write(f"      P(mediana_random <= mediana_obs): {r['p_median']}  successi={r['success_median']}\n")
            f.write(f"      P(count_random >= count_obs):     {r['p_count']}  successi={r['success_count']}\n")

        f.write("\nNota metodologica:\n")
        f.write("  I p-value valgono solo rispetto ai modelli nulli implementati.\n")
        f.write("  Un p-value pari a zero in simulazione viene riportato come limite superiore p < 1/N.\n")
        f.write("  Il test non costituisce una dimostrazione della congettura.\n")
        f.write("  La denominazione ORO/ARGENTO è una classificazione operativa del dataset, non una prova fisica.\n")
        f.write("  Lo scan M0 è da intendersi come controllo esplorativo separato dal test principale con M0=14.\n")


# =============================================================================
# 11. GRAFICI
# =============================================================================

def plot_mc_distribution(values, observed, title, xlabel, outname):
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, alpha=0.75, edgecolor="black")
    plt.axvline(observed, linestyle="--", linewidth=2.5, label=f"Osservato = {observed:.4g}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Numero simulazioni")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outname, dpi=150)
    plt.close()


def plot_individual(obs, dataset_name, outname):
    labels = obs["labels"]
    devs = obs["deviations"] * 100
    plt.figure(figsize=(12, 8))
    plt.barh(labels, devs)
    plt.axvline(THRESHOLD * 100, linestyle="--", linewidth=2, label=f"Soglia {THRESHOLD*100:.3f}%")
    plt.xlabel("Scarto relativo (%)")
    plt.title(f"RHSIC V0013 - Scarti individuali ({dataset_name})")
    plt.legend()
    plt.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(outname, dpi=150)
    plt.close()


# =============================================================================
# 12. MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("RHSIC - Monte Carlo V0013bb (ORO vs ARGENTO)")
    print("=" * 80)

    # Caricamento zeri
    zeros_file = find_zeros_file()
    if zeros_file is None:
        raise FileNotFoundError("Nessun file zeri trovato. Metti zeros_50000.txt, zeros_5000.txt o zeros_5000.csv nella stessa cartella.")

    zeros = load_zeros_any(zeros_file, ZERO_LIMIT)
    if len(zeros) == 0:
        raise RuntimeError("Nessuno zero caricato.")

    print(f"File zeri: {zeros_file}")
    print(f"Zeri caricati: {len(zeros)}")
    print(f"Intervallo zeri: {zeros[0]:.6f} - {zeros[-1]:.6f}")
    print(f"Sistemi ORO: {len(ORO_DATA)}")
    print(f"Sistemi ARGENTO: {len(ARGENTO_DATA)}")
    print(f"Sistemi TOTALI: {len(TOTAL_DATA)}")

    start = time.time()

    # ---- TEST SU ORO ----
    print("\n" + "=" * 80)
    print("ANALISI ORO (14 sistemi originali)")
    print("=" * 80)
    obs_oro, results_oro = run_main_tests(ORO_DATA, "ORO", zeros)
    write_individual_results("rhsic_v0013_ORO_individual_results.csv", ORO_DATA, obs_oro, "ORO")
    plot_individual(obs_oro, "ORO", "rhsic_v0013_ORO_individual_deviations.png")
    loo_oro = run_leave_one_out(ORO_DATA, "ORO", zeros)
    class_oro = run_class_tests(ORO_DATA, "ORO", zeros)
    m0_oro = run_m0_scan(ORO_DATA, "ORO", zeros)
    write_csv("rhsic_v0013_ORO_leave_one_out.csv", loo_oro)
    write_csv("rhsic_v0013_ORO_class_tests.csv", class_oro)
    write_csv("rhsic_v0013_ORO_m0_scan.csv", m0_oro)

    # ---- TEST SU ARGENTO ----
    print("\n" + "=" * 80)
    print("ANALISI ARGENTO (26 sistemi aggiuntivi)")
    print("=" * 80)
    obs_argento, results_argento = run_main_tests(ARGENTO_DATA, "ARGENTO", zeros)
    write_individual_results("rhsic_v0013_ARGENTO_individual_results.csv", ARGENTO_DATA, obs_argento, "ARGENTO")
    plot_individual(obs_argento, "ARGENTO", "rhsic_v0013_ARGENTO_individual_deviations.png")
    loo_argento = run_leave_one_out(ARGENTO_DATA, "ARGENTO", zeros)
    class_argento = run_class_tests(ARGENTO_DATA, "ARGENTO", zeros)
    m0_argento = run_m0_scan(ARGENTO_DATA, "ARGENTO", zeros)
    write_csv("rhsic_v0013_ARGENTO_leave_one_out.csv", loo_argento)
    write_csv("rhsic_v0013_ARGENTO_class_tests.csv", class_argento)
    write_csv("rhsic_v0013_ARGENTO_m0_scan.csv", m0_argento)

    # ---- SUMMARY ----
    write_summary("rhsic_v0013_summary.txt",
                  obs_oro, results_oro,
                  obs_argento, results_argento,
                  loo_oro, loo_argento,
                  class_oro, class_argento)

    # ---- OUTPUT FINALE ----
    print("\n" + "=" * 80)
    print("RISULTATI CONFRONTO ORO vs ARGENTO")
    print("=" * 80)

    print(f"\nORO ({len(ORO_DATA)} sistemi):")
    print(f"  Media scarti:   {obs_oro['mean_dev']*100:.6f}%")
    print(f"  Mediana scarti: {obs_oro['median_dev']*100:.6f}%")
    print(f"  Count < soglia: {obs_oro['count_below']} / {len(ORO_DATA)}")
    for r in results_oro:
        print(f"    {r['test']}: p_mean={r['p_mean']}")

    print(f"\nARGENTO ({len(ARGENTO_DATA)} sistemi):")
    print(f"  Media scarti:   {obs_argento['mean_dev']*100:.6f}%")
    print(f"  Mediana scarti: {obs_argento['median_dev']*100:.6f}%")
    print(f"  Count < soglia: {obs_argento['count_below']} / {len(ARGENTO_DATA)}")
    for r in results_argento:
        print(f"    {r['test']}: p_mean={r['p_mean']}")

    print("\nFile generati:")
    print("  rhsic_v0013_summary.txt")
    print("  rhsic_v0013_ORO_*.csv / .png")
    print("  rhsic_v0013_ARGENTO_*.csv / .png")
    print(f"\nTempo totale: {time.time() - start:.1f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()