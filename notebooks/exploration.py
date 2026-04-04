"""
SenSante - Exploration du dataset patients_dakar.csv
Lab 1 : Git, Python et Structure Projet
"""

import pandas as pd
import matplotlib.pyplot as plt

# ===== CHARGER LES DONNEES =====
df = pd.read_csv("data/patients_dakar.csv")

# ===== PREMIERS APERCUS =====
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)

print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")

print(f"\n--- 5 premiers patients ---")
print(df.head())

# ===== STATISTIQUES DE BASE =====
print(f"\n--- Statistiques descriptives ---")
print(df.describe().round(2))

# ===== REPARTITION DES DIAGNOSTICS =====
print(f"\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f"  {diag:12s} : {count:3d} patients ({pct:.1f}%)")

# ===== REPARTITION PAR REGION =====
print(f"\n--- Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print(f"  {region:15s} : {count:3d} patients")

# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print(f"\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f"  {diag:12s} : {temp:.1f} C")

# ===== EXERCICE 1 =====
print(f"\n--- Patients par sexe et diagnostic ---")
sexe_diag = df.groupby(["sexe", "diagnostic"]).size()
print(sexe_diag)

print(f"\n{'=' * 50}")
print("Exploration terminee !")
print("Prochain lab : entrainer un modele ML")
print(f"{'=' * 50}")

# ===== VISUALISATIONS =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Apercu du dataset SenSante", fontsize=14, fontweight='bold')

# Graphique 1 : Diagnostics
diag_counts.plot(kind='bar', ax=axes[0], color=['green', 'red', 'blue', 'orange'])
axes[0].set_title("Diagnostics")
axes[0].set_ylabel("Patients")
axes[0].tick_params(axis='x', rotation=30)

# Graphique 2 : Temperatures par diagnostic
for diag in df["diagnostic"].unique():
    subset = df[df["diagnostic"] == diag]["temperature"]
    axes[1].hist(subset, alpha=0.5, label=diag, bins=15)
axes[1].set_title("Temperature par diagnostic")
axes[1].set_xlabel("Temperature (C)")
axes[1].legend(fontsize=8)

# Graphique 3 : Top 5 regions
top5 = df["region"].value_counts().head(5)
top5.plot(kind='barh', ax=axes[2], color='steelblue')
axes[2].set_title("Top 5 regions")

plt.tight_layout()
plt.savefig("notebooks/visualisation_dataset.png", dpi=150, bbox_inches='tight')
print("\nGraphique sauvegarde : notebooks/visualisation_dataset.png")
plt.show()
