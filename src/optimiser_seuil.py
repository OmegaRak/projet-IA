
"""Optimisation du seuil de décision du Gradient Boosting."""

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from train_mlflow import (
    BASE_DIR,
    FEATURES,
    CIBLE,
    COLONNES_CATEGORIELLES,
    charger_donnees,
)


def main():

    # 1. Chargement des données
    chemin = BASE_DIR / "data" / "Livraisons.csv"
    df = charger_donnees(chemin)

    X = df[FEATURES].copy()
    y = df[CIBLE].copy()

    # 2. Encodage
    for col in COLONNES_CATEGORIELLES:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # 3. Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Modèle optimisé
    modele = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=10,
        random_state=42,
    )

    poids = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    modele.fit(X_train, y_train, sample_weight=poids)

    # 5. Probabilités de prédiction
    probabilites = modele.predict_proba(X_test)[:, 1]

    # 6. Test de plusieurs seuils
    seuils = [0.3, 0.4, 0.5, 0.6, 0.7]
    resultats = []

    print("\n========== COMPARAISON DES SEUILS ==========")

    for seuil in seuils:

        y_pred = (probabilites >= seuil).astype(int)

        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )

        matrice = confusion_matrix(y_test, y_pred)

        resultats.append({
            "seuil": seuil,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "vrais_negatifs": matrice[0, 0],
            "faux_positifs": matrice[0, 1],
            "faux_negatifs": matrice[1, 0],
            "vrais_positifs": matrice[1, 1],
        })

        print(f"\nSeuil : {seuil}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"Faux négatifs : {matrice[1, 0]}")

    # 7. Sauvegarde
    tableau = pd.DataFrame(resultats)

    dossier = BASE_DIR / "reports"
    dossier.mkdir(exist_ok=True)

    tableau.to_csv(
        dossier / "comparaison_seuils.csv",
        index=False,
    )

    print("\n========== TABLEAU FINAL ==========")
    print(tableau.round(4).to_string(index=False))

    print("\nRésultats sauvegardés dans reports/comparaison_seuils.csv")


if __name__ == "__main__":
    main()
