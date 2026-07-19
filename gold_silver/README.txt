RHSIC V0013b - Controlled GOLD / SILVER Validation

Contents:
- TestMontecarlo_RHSIC_V0013b_controllata.py

Changes compared with the previous version:

1. Corrected loading of Riemann zeros:
   the parser now reads the file line by line and, for two-column files,
   automatically uses the last value of each line as the zero.
   This prevents row indices (1, 2, 3, ...) from being incorrectly interpreted
   as Riemann zeros.

2. Neutral wording for the GOLD dataset:
   "14 original systems / predefined reference sample".

3. Added methodological note in the summary:
   GOLD/SILVER is an operational dataset classification only.
   The M0 scan is provided as an exploratory analysis and is independent
   from the primary validation test.

Usage:

1. Copy either:
      zeros_50000.txt
   or
      zeros_5000.txt

   into the same folder as the Python script.

2. Run:

   python TestMontecarlo_RHSIC_V0013b_controllata.py

Main outputs:

- rhsic_v0013_summary.txt
- rhsic_v0013_ORO_individual_results.csv
- rhsic_v0013_ORO_leave_one_out.csv
- rhsic_v0013_ORO_class_tests.csv
- rhsic_v0013_ORO_m0_scan.csv
- rhsic_v0013_ARGENTO_individual_results.csv
- rhsic_v0013_ARGENTO_leave_one_out.csv
- rhsic_v0013_ARGENTO_class_tests.csv
- rhsic_v0013_ARGENTO_m0_scan.csv
- PNG figures for both GOLD and SILVER datasets



RHSIC V0013b - ORO / ARGENTO controllata

Contenuto:
- TestMontecarlo_RHSIC_V0013b_controllata.py

Differenze rispetto al sorgente caricato:
1. Corretto il caricamento degli zeri:
   il parser ora legge per riga e, nei file a due colonne, usa l'ultimo numero
   della riga come valore dello zero. Questo evita di interpretare gli indici
   1, 2, 3, ... come falsi zeri.
2. Reso più neutro il commento sul campione ORO:
   "14 sistemi originali / campione base predefinito".
3. Aggiunta nota metodologica nel summary:
   ORO/ARGENTO è una classificazione operativa; lo scan M0 è esplorativo.

Uso:
1. Copiare nella stessa cartella il file zeros_50000.txt oppure zeros_5000.txt.
2. Eseguire:
   python TestMontecarlo_RHSIC_V0013b_controllata.py

Output principali:
- rhsic_v0013_summary.txt
- rhsic_v0013_ORO_individual_results.csv
- rhsic_v0013_ORO_leave_one_out.csv
- rhsic_v0013_ORO_class_tests.csv
- rhsic_v0013_ORO_m0_scan.csv
- rhsic_v0013_ARGENTO_individual_results.csv
- rhsic_v0013_ARGENTO_leave_one_out.csv
- rhsic_v0013_ARGENTO_class_tests.csv
- rhsic_v0013_ARGENTO_m0_scan.csv
- grafici PNG ORO e ARGENTO
