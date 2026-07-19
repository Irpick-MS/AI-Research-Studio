#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RHSIC Monte Carlo V1.0 - 100.000 iterazioni

Versione congelabile per test preliminare sui 14 sistemi favorevoli.
Esegue automaticamente:
  1) Monte Carlo uniform in scala logaritmica delle frequenze
  2) Monte Carlo permutation delle frequenze osservate

Output generati:
  - monte_carlo_uniform_results.txt
  - monte_carlo_permutation_results.txt
  - monte_carlo_uniform_100000.csv
  - monte_carlo_permutation_100000.csv
  - monte_carlo_uniform_plot.png
  - monte_carlo_permutation_plot.png

Requisiti:
  - numpy
  - matplotlib
  - file zeri nella stessa cartella dello script:
      zeros_50000.txt oppure zeros_5000.txt oppure zeros.txt oppure zetazero.txt oppure zeros1.txt
"""

import csv
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. DATI REALI: 14 SISTEMI
# =============================================================================

REAL_DATA = [
    (14.0, 67.0, 'GRS 1915+105', 'stellar'),
    (6.3, 300.0, 'GRO J1655-40 (300)', 'stellar'),
    (6.3, 451.0, 'GRO J1655-40 (451)', 'stellar'),
    (9.1, 184.0, 'XTE J1550-564 (184)', 'stellar'),
    (9.1, 276.0, 'XTE J1550-564 (276)', 'stellar'),
    (7.8, 227.5, 'XTE J1859+226', 'stellar'),
    (428.0, 3.3, 'M82 X-1 (3.3)', 'imbh'),
    (428.0, 5.0, 'M82 X-1 (5.0)', 'imbh'),
    (5000.0, 0.30, 'NGC 1313 X-1 (0.30)', 'imbh'),
    (5000.0, 0.45, 'NGC 1313 X-1 (0.45)', 'imbh'),
    (1.7e6, 2.68e-4, 'RE J1034+396', 'agn'),
    (62.0, 264.0, 'GW150914', 'gw'),
    (20.8, 265.0, 'GW151226', 'gw'),
    (332.0, 62.0, 'GW190521', 'gw')
]


# =============================================================================
# 2. PARAMETRI CONGELATI V1.0
# =============================================================================

LIMIT_ZERO = 2500
N_ITER = 100000
THRESHOLD = 0.002       # 0.2%
SEED = 42
ZERO_FILE_CANDIDATES = [
    'zeros_50000.txt',
    'zeros_5000.txt',
    'zeros.txt',
    'zetazero.txt',
    'zeros1.txt'
]


# =============================================================================
# 3. CARICAMENTO ZERI
# =============================================================================

def find_zeros_file():
    """Cerca automaticamente il file degli zeri nella cartella corrente."""
    for name in ZERO_FILE_CANDIDATES:
        if os.path.isfile(name):
            return name
    return None


def load_zeros_any(filename, limit=2500):
    """
    Legge un file di zeri in formato testo.
    Supporta:
      - formato semplice: indice valore
      - formato singolo valore per riga
      - formato mpf('...')
    """
    zeros = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        pattern = r"mpf\('([\d.]+)'\)|([\d]+\.\d+)"
        matches = re.findall(pattern, content)

        if matches:
            for m in matches:
                val = m[0] if m[0] else m[1]
                try:
                    t = float(val)
                    if t <= limit:
                        zeros.append(t)
                except ValueError:
                    pass
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    candidate = parts[1] if len(parts) >= 2 else parts[0]
                    try:
                        t = float(candidate)
                        if t <= limit:
                            zeros.append(t)
                    except ValueError:
                        continue

        zeros = np.array(sorted(set(zeros)), dtype=float)
        print(f"    Caricati {len(zeros)} zeri da '{filename}' con t <= {limit}")
        return zeros

    except FileNotFoundError:
        print(f"    [ERRORE] File '{filename}' non trovato.")
        return np.array([])
    except Exception as e:
        print(f"    [ERRORE] Lettura fallita: {e}")
        return np.array([])


# =============================================================================
# 4. FUNZIONI DI CALCOLO
# =============================================================================

def nearest_zero(phi, zeros):
    """Trova lo zero più vicino a phi usando ricerca binaria."""
    if len(zeros) == 0:
        return np.nan

    idx = np.searchsorted(zeros, phi)
    if idx == 0:
        return zeros[0]
    if idx == len(zeros):
        return zeros[-1]

    left = zeros[idx - 1]
    right = zeros[idx]
    return left if abs(phi - left) <= abs(right - phi) else right


def relative_deviation(phi, zeros):
    """Scarto relativo dallo zero più vicino."""
    t_n = nearest_zero(phi, zeros)
    if not np.isfinite(t_n) or t_n == 0:
        return np.nan
    return abs(phi - t_n) / t_n


def compute_observed_stats(selected, zeros):
    """Calcola le statistiche osservate sui dati reali."""
    phis = []
    nearest = []
    deviations = []

    for mass, freq, name, cls in selected:
        phi = freq * mass / 14.0
        zero = nearest_zero(phi, zeros)
        dev = relative_deviation(phi, zeros)
        phis.append(phi)
        nearest.append(zero)
        deviations.append(dev)

    deviations = np.array(deviations, dtype=float)

    return {
        'phis': np.array(phis, dtype=float),
        'nearest_zeros': np.array(nearest, dtype=float),
        'deviations': deviations,
        'mean_dev': float(np.nanmean(deviations)),
        'median_dev': float(np.nanmedian(deviations)),
        'count_below': int(np.sum(deviations < THRESHOLD)),
        'labels': [d[2] for d in selected],
        'classes': [d[3] for d in selected]
    }


def z_score_observed(obs_value, mc_values, direction='lower'):
    """
    Calcola Z-score rispetto alla distribuzione Monte Carlo.
    direction='lower': valori osservati bassi sono più estremi, es. media/mediana scarti.
    direction='higher': valori osservati alti sono più estremi, es. conteggio match.
    """
    mu = float(np.mean(mc_values))
    sigma = float(np.std(mc_values, ddof=1))
    if sigma == 0:
        return np.nan
    z = (obs_value - mu) / sigma
    return z if direction == 'lower' else -z


# =============================================================================
# 5. MONTE CARLO
# =============================================================================

def monte_carlo_test(selected, zeros, n_iter, test_type, seed, threshold):
    """Esegue il test Monte Carlo richiesto."""
    rng = np.random.default_rng(seed)

    n_sys = len(selected)
    masses = np.array([d[0] for d in selected], dtype=float)
    freqs_real = np.array([d[1] for d in selected], dtype=float)

    log_freq_min = np.log10(np.min(freqs_real) * 0.5)
    log_freq_max = np.log10(np.max(freqs_real) * 2.0)

    mean_devs = np.zeros(n_iter, dtype=float)
    median_devs = np.zeros(n_iter, dtype=float)
    count_below = np.zeros(n_iter, dtype=int)

    print(f"    Esecuzione {test_type}: {n_iter} iterazioni...")
    start = time.time()

    for i in range(n_iter):
        if test_type == 'uniform':
            log_f = rng.uniform(log_freq_min, log_freq_max, size=n_sys)
            f_rand = 10.0 ** log_f
        elif test_type == 'permutation':
            f_rand = rng.permutation(freqs_real)
        else:
            raise ValueError("test_type deve essere 'uniform' oppure 'permutation'")

        devs = np.array([
            relative_deviation(f_rand[j] * masses[j] / 14.0, zeros)
            for j in range(n_sys)
        ], dtype=float)

        mean_devs[i] = np.nanmean(devs)
        median_devs[i] = np.nanmedian(devs)
        count_below[i] = int(np.sum(devs < threshold))

        if (i + 1) % max(1, n_iter // 10) == 0:
            pct = 100 * (i + 1) // n_iter
            elapsed = time.time() - start
            print(f"      {pct}% completato ({elapsed:.1f}s)")

    elapsed = time.time() - start
    print(f"    Completato {test_type} in {elapsed:.1f}s")

    return {
        'mean_devs': mean_devs,
        'median_devs': median_devs,
        'count_below': count_below,
        'elapsed_seconds': elapsed
    }


# =============================================================================
# 6. SALVATAGGIO CSV E REPORT
# =============================================================================

def save_mc_csv(mc, test_type):
    """Salva tutti i 100.000 dati simulati in CSV."""
    filename = f"monte_carlo_{test_type}_{len(mc['mean_devs'])}.csv"
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['iteration', 'mean_deviation', 'median_deviation', 'count_below_threshold'])
        for i, (m, med, c) in enumerate(zip(mc['mean_devs'], mc['median_devs'], mc['count_below']), start=1):
            writer.writerow([i, f"{m:.12g}", f"{med:.12g}", int(c)])
    print(f"    CSV salvato: {filename}")
    return filename


def save_observed_csv(obs):
    """Salva i dati osservati e gli scarti individuali."""
    filename = 'rhsic_observed_14_systems.csv'
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['system', 'class', 'phi', 'nearest_zero', 'relative_deviation', 'relative_deviation_percent'])
        for name, cls, phi, z, dev in zip(obs['labels'], obs['classes'], obs['phis'], obs['nearest_zeros'], obs['deviations']):
            writer.writerow([name, cls, f"{phi:.12g}", f"{z:.12g}", f"{dev:.12g}", f"{dev * 100:.8f}"])
    print(f"    CSV osservati salvato: {filename}")
    return filename


def print_and_save_results(obs, mc, n_iter, threshold, test_type, zero_filename, zero_count, mc_csv_name):
    """Stampa e salva il report testuale."""
    p_mean = float(np.mean(mc['mean_devs'] <= obs['mean_dev']))
    p_median = float(np.mean(mc['median_devs'] <= obs['median_dev']))
    p_count = float(np.mean(mc['count_below'] >= obs['count_below']))

    z_mean = z_score_observed(obs['mean_dev'], mc['mean_devs'], direction='lower')
    z_median = z_score_observed(obs['median_dev'], mc['median_devs'], direction='lower')
    z_count = z_score_observed(obs['count_below'], mc['count_below'], direction='higher')

    report_file = f"monte_carlo_{test_type}_results.txt"

    lines = []
    lines.append("=" * 78)
    lines.append(f"RHSIC MONTE CARLO V1.0 - TEST {test_type.upper()}")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Data esecuzione: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"File zeri: {zero_filename}")
    lines.append(f"Numero zeri caricati: {zero_count}")
    lines.append(f"Limite superiore zeri: {LIMIT_ZERO}")
    lines.append(f"Sistemi analizzati: {len(obs['labels'])}")
    lines.append(f"Test: {test_type}")
    lines.append(f"Iterazioni: {n_iter}")
    lines.append(f"Soglia match: {threshold * 100:.4f}%")
    lines.append(f"Seed: {SEED}")
    lines.append(f"CSV dati Monte Carlo: {mc_csv_name}")
    lines.append(f"Tempo esecuzione: {mc['elapsed_seconds']:.2f} s")
    lines.append("")
    lines.append("IPOTESI NULLA")
    if test_type == 'uniform':
        lines.append("Le frequenze sono campionate casualmente in scala logaritmica nel range")
        lines.append("[0.5 x frequenza minima osservata, 2 x frequenza massima osservata].")
    else:
        lines.append("Le frequenze osservate vengono permutate casualmente tra le masse reali.")
    lines.append("")
    lines.append("FORMULA")
    lines.append("Phi = f * M / 14")
    lines.append("Delta = |Phi - t_n| / t_n, dove t_n è lo zero più vicino.")
    lines.append("")
    lines.append("STATISTICHE OSSERVATE")
    lines.append(f"Media scarti: {obs['mean_dev'] * 100:.8f}%")
    lines.append(f"Mediana scarti: {obs['median_dev'] * 100:.8f}%")
    lines.append(f"Conteggio Delta < {threshold * 100:.4f}%: {obs['count_below']} / {len(obs['labels'])}")
    lines.append("")
    lines.append("P-VALUE")
    lines.append(f"P(media_MC <= media_osservata): {p_mean:.10f}")
    lines.append(f"P(mediana_MC <= mediana_osservata): {p_median:.10f}")
    lines.append(f"P(conteggio_MC >= conteggio_osservato): {p_count:.10f}")
    lines.append("")
    lines.append("Z-SCORE")
    lines.append(f"Z media scarti: {z_mean:.6f}")
    lines.append(f"Z mediana scarti: {z_median:.6f}")
    lines.append(f"Z conteggio: {z_count:.6f}")
    lines.append("")
    lines.append("SCARTI INDIVIDUALI")
    for phi, zero, dev, name in zip(obs['phis'], obs['nearest_zeros'], obs['deviations'], obs['labels']):
        lines.append(f"{name:30s} | Phi={phi:14.6f} | zero={zero:14.6f} | Delta={dev * 100:10.6f}%")
    lines.append("")
    lines.append("DATI UTILIZZATI")
    lines.append("Sistemi osservativi: 14 sistemi astrofisici inseriti nel sorgente REAL_DATA.")
    lines.append(f"Zeri della funzione zeta: {zero_filename}")
    lines.append("Il file degli zeri deve essere conservato insieme a questo report per riproducibilità.")
    lines.append("")
    lines.append("GIUDIZIO AUTOMATICO")
    if p_mean < 0.01:
        lines.append("La statistica della media risulta significativa a p < 0.01.")
    else:
        lines.append("La statistica della media NON risulta significativa a p < 0.01.")

    text = "\n".join(lines)
    print("\n" + text + "\n")

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"    Report salvato: {report_file}")
    return report_file


# =============================================================================
# 7. GRAFICO
# =============================================================================

def generate_plot(obs, mc, threshold, test_type):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.hist(mc['mean_devs'] * 100, bins=60, alpha=0.75, edgecolor='black')
    ax1.axvline(obs['mean_dev'] * 100, linestyle='--', linewidth=2.5,
                label=f"Osservato = {obs['mean_dev'] * 100:.6f}%")
    ax1.set_xlabel('Scarto medio (%)')
    ax1.set_ylabel('Frequenza')
    ax1.set_title(f'Distribuzione media scarti - {test_type}')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    bins2 = np.arange(-0.5, int(max(mc['count_below'])) + 1.5, 1)
    ax2.hist(mc['count_below'], bins=bins2, alpha=0.75, edgecolor='black', density=True)
    ax2.axvline(obs['count_below'], linestyle='--', linewidth=2.5,
                label=f"Osservato = {obs['count_below']}")
    ax2.set_xlabel(f'Numero di match sotto soglia ({threshold * 100:.3f}%)')
    ax2.set_ylabel('Densità')
    ax2.set_title(f'Distribuzione conteggio match - {test_type}')
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    colors = ['green' if d < threshold else 'orange' if d < 0.005 else 'red' for d in obs['deviations']]
    ax3.barh(obs['labels'], obs['deviations'] * 100, color=colors, edgecolor='black')
    ax3.axvline(threshold * 100, linestyle='--', linewidth=1.5,
                label=f'Soglia {threshold * 100:.3f}%')
    ax3.set_xlabel('Scarto (%)')
    ax3.set_title('Scarti individuali osservati')
    ax3.legend()
    ax3.grid(alpha=0.3, axis='x')

    ax4 = axes[1, 1]
    ax4.scatter(obs['phis'], np.zeros_like(obs['phis']), s=80, zorder=5, label='Phi osservati')
    ax4.set_xlabel('Phi = f * M / 14')
    ax4.set_yticks([])
    ax4.set_title('Posizione dei Phi osservati')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    filename = f"monte_carlo_{test_type}_plot.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"    Grafico salvato: {filename}")
    return filename


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    print("\n" + "=" * 78)
    print("RHSIC MONTE CARLO V1.0 - 100.000 ITERAZIONI")
    print("=" * 78)
    print(f"Sistemi: {len(REAL_DATA)}")
    print(f"Iterazioni per test: {N_ITER}")
    print(f"Soglia: {THRESHOLD * 100:.4f}%")
    print(f"Seed base: {SEED}")

    print("\n[1] Ricerca file zeri...")
    zero_filename = find_zeros_file()
    if zero_filename is None:
        print("[ERRORE] Nessun file zeri trovato nella cartella dello script.")
        print("Inserire uno di questi file:")
        for name in ZERO_FILE_CANDIDATES:
            print(f"  - {name}")
        sys.exit(1)

    print(f"    Trovato: {zero_filename}")

    print("\n[2] Caricamento zeri...")
    zeros = load_zeros_any(zero_filename, LIMIT_ZERO)
    if len(zeros) == 0:
        print("[ERRORE] Nessuno zero caricato.")
        sys.exit(1)

    print(f"    Intervallo zeri: {zeros[0]:.6f} - {zeros[-1]:.6f}")

    print("\n[3] Calcolo statistiche osservate...")
    selected = REAL_DATA
    obs = compute_observed_stats(selected, zeros)
    save_observed_csv(obs)

    print(f"    Media scarti osservata: {obs['mean_dev'] * 100:.8f}%")
    print(f"    Mediana scarti osservata: {obs['median_dev'] * 100:.8f}%")
    print(f"    Match sotto soglia: {obs['count_below']} / {len(selected)}")

    for offset, test_type in enumerate(['uniform', 'permutation']):
        print("\n" + "-" * 78)
        print(f"[4] Test Monte Carlo: {test_type}")
        print("-" * 78)

        mc = monte_carlo_test(
            selected=selected,
            zeros=zeros,
            n_iter=N_ITER,
            test_type=test_type,
            seed=SEED + offset,
            threshold=THRESHOLD
        )

        mc_csv = save_mc_csv(mc, test_type)
        print_and_save_results(
            obs=obs,
            mc=mc,
            n_iter=N_ITER,
            threshold=THRESHOLD,
            test_type=test_type,
            zero_filename=zero_filename,
            zero_count=len(zeros),
            mc_csv_name=mc_csv
        )
        try:
            generate_plot(obs, mc, THRESHOLD, test_type)
        except Exception as e:
            print(f"    [AVVISO] Grafico non generato per {test_type}: {e}")

    print("\n" + "=" * 78)
    print("TEST COMPLETATO")
    print("File principali generati:")
    print("  - monte_carlo_uniform_100000.csv")
    print("  - monte_carlo_permutation_100000.csv")
    print("  - monte_carlo_uniform_results.txt")
    print("  - monte_carlo_permutation_results.txt")
    print("  - rhsic_observed_14_systems.csv")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
