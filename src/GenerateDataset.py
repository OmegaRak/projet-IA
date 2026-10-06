import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

nombreCommandes = 15000

# Transporteurs / sociétés de livraison
transporteurs = [
    "Trans Mahefa",
    "Sonatra",
    "Mami",
    "Mada Trans",
    "Service Rapide"
]

# Probabilité de retard de base par transporteur
fiabiliteTransporteur = {
    "Trans Mahefa": 0.10,
    "Sonatra": 0.20,
    "Mami": 0.16,
    "Mada Trans": 0.07,
    "Service Rapide": 0.28,
}

# Modes d'expédition
modesExpedition = [
    "Standard",
    "Express",
    "Economique",
    "Prioritaire"
]

# Délai de base selon le mode d'expédition
delaiBaseMode = {
    "Standard": 5,
    "Express": 2,
    "Economique": 9,
    "Prioritaire": 1
}

# Catégories de produits
categoriesProduit = [
    "Electronique",
    "Vetements",
    "Produits agricoles",
    "Alimentaire",
    "Artisanat",
    "Pieces detachees",
    "Materiel medical"
]

# Grandes villes / régions de Madagascar
# Distance approximative depuis Antananarivo
villesDistance = {
    "Antananarivo": 0,
    "Antsirabe": 170,
    "Fianarantsoa": 410,
    "Toamasina": 350,
    "Mahajanga": 560,
    "Toliara": 930,
    "Antsiranana": 1100,
    "Morondava": 660,
    "Ambatondrazaka": 240,
    "Fenerive Est": 430,
}

regions = list(villesDistance.keys())

# Dates de commande sur 2 ans
dateDebut = datetime(2024, 1, 1)

datesCommande = [
    dateDebut + timedelta(days=int(x))
    for x in np.random.randint(0, 730, nombreCommandes)
]

# Construction du dataset
regionChoisie = np.random.choice(
    regions,
    nombreCommandes,
    p=[
        0.30, 0.12, 0.10, 0.13, 0.08, 0.08, 0.06, 0.05, 0.04, 0.04
    ]
)


df = pd.DataFrame({

    "idCommande": range(1, nombreCommandes + 1),

    "dateCommande": datesCommande,

    "transporteur": np.random.choice(
        transporteurs,
        nombreCommandes,
        p=[0.25, 0.20, 0.20, 0.20, 0.15]
    ),

    "modeExpedition": np.random.choice(
        modesExpedition,
        nombreCommandes,
        p=[0.40, 0.20, 0.25, 0.15]
    ),

    "categorieProduit": np.random.choice(
        categoriesProduit,
        nombreCommandes
    ),

    "region": regionChoisie,

    "quantite": np.random.randint(
        1,
        10,
        nombreCommandes
    ),
})

# Calcul de la distance
# Distance de base selon la région
distanceBase = df["region"].map(villesDistance)


# Ajout d'une petite variation locale réaliste
df["distanceKm"] = np.round(
    distanceBase
    + np.abs(np.random.normal(5, 8, nombreCommandes)),
    1
)

# Calcul de la date prévue
delaiPrevuJours = (
    df["modeExpedition"].map(delaiBaseMode)
    + (df["distanceKm"] / 300).round().astype(int)
)

delaiPrevuJours = delaiPrevuJours.clip(lower=1)


df["datePrevue"] = (
    df["dateCommande"]
    + pd.to_timedelta(delaiPrevuJours, unit="D")
)

# Génération du retard
mois = df["dateCommande"].dt.month

jourSemaine = df["dateCommande"].dt.day_name()


# Saison des pluies à Madagascar
saisonPluies = mois.isin(
    [11, 12, 1, 2, 3]
).astype(int)


# Régions considérées comme plus difficiles d'accès
regionsEnclavees = df["region"].isin([
    "Toliara",
    "Antsiranana",
    "Morondava",
    "Fenerive Est"
]).astype(int)

# Calcul de la probabilité de retard
probaRetard = (
    df["transporteur"].map(fiabiliteTransporteur) * 1.4

    + (df["modeExpedition"] == "Economique") * 0.12

    + (df["modeExpedition"] == "Express") * -0.05

    + (df["distanceKm"] > 500) * 0.12

    + regionsEnclavees * 0.10

    + saisonPluies * 0.18

    + jourSemaine.isin(
        ["Friday", "Saturday"]
    ) * 0.05

    + np.random.normal(
        0,
        0.05,
        nombreCommandes
    )
)


# Limiter la probabilité entre 1% et 95%
probaRetard = probaRetard.clip(
    0.01,
    0.95
)

# Création de la variable cible : retard

df["retard"] = (
    np.random.uniform(
        0,
        1,
        nombreCommandes
    ) < probaRetard
).astype(int)

# Format final des dates
df["dateCommande"] = df[
    "dateCommande"
].dt.strftime("%Y-%m-%d")


df["datePrevue"] = df[
    "datePrevue"
].dt.strftime("%Y-%m-%d")


colonnesFinales = [

    "idCommande", "dateCommande", "datePrevue", 
    "transporteur", "modeExpedition", "region", 
    "categorieProduit","quantite","distanceKm","retard"
]
df = df[colonnesFinales]

# Sauvegarde du dataset
dossierSortie = "data"

os.makedirs(
    dossierSortie,
    exist_ok=True
)


cheminSortie = os.path.join(
    dossierSortie,
    "Livraisons.csv"
)


df.to_csv(
    cheminSortie,
    index=False
)

# Affichage des résultats
print(
    f"Dataset généré : "
    f"{len(df)} lignes -> {cheminSortie}"
)

print(
    f"Taux de retard global : "
    f"{df['retard'].mean() * 100:.1f}%"
)

print("\nAperçu :")

print(
    df.head()
)

print("\nTaux de retard par région :")

print(
    df.groupby("region")["retard"]
    .mean()
    .sort_values(ascending=False)
    .round(3)
)


print("\nTaux de retard par transporteur :")

print(
    df.groupby("transporteur")["retard"]
    .mean()
    .sort_values(ascending=False)
    .round(3)
)