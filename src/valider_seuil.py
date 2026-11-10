
"""Validation du seuil et évaluation finale du modèle."""

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
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

    # 2. Encodage des variables catégorielles
    for col in COLONNES_CATEGORIELLES:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # 3. Séparation : 60 % train, 20 % validation, 20 % test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=123,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=123,
        stratify=y_temp,
    )

    print("\n========== RÉPARTITION DES DONNÉES ==========")
    print("Train      :", len(X_train))
    print("Validation :", len(X_val))
    print("Test       :", len(X_test))

    # 4. Entraînement sur le jeu Train uniquement
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

    modele.fit(
        X_train,
        y_train,
        sample_weight=poids,
    )

    # 5. Recherche du seuil sur Validation
    probabilites_val = modele.predict_proba(X_val)[:, 1]

    seuils = [0.3, 0.4, 0.5, 0.6, 0.7]
    resultats = []

    for seuil in seuils:

        pred_val = (probabilites_val >= seuil).astype(int)

        resultats.append({
            "seuil": seuil,
            "precision": precision_score(
                y_val, pred_val, zero_division=0
            ),
            "recall": recall_score(
                y_val, pred_val, zero_division=0
            ),
            "f1_score": f1_score(
                y_val, pred_val, zero_division=0
            ),
        })

    tableau = pd.DataFrame(resultats)

    print("\n========== RÉSULTATS VALIDATION ==========")
    print(tableau.round(4).to_string(index=False))

    meilleur_seuil = tableau.loc[
        tableau["f1_score"].idxmax(), "seuil"
    ]

    print("\nSeuil choisi sur Validation :", meilleur_seuil)

    # 6. Évaluation finale sur Test
    probabilites_test = modele.predict_proba(X_test)[:, 1]
    pred_test = (probabilites_test >= meilleur_seuil).astype(int)

    matrice = confusion_matrix(y_test, pred_test)

    print("\n========== ÉVALUATION FINALE TEST ==========")
    print("Seuil utilisé :", meilleur_seuil)
    print("Accuracy      :", round(accuracy_score(y_test, pred_test), 4))
    print("Precision     :", round(precision_score(y_test, pred_test, zero_division=0), 4))
    print("Recall        :", round(recall_score(y_test, pred_test, zero_division=0), 4))
    print("F1-score      :", round(f1_score(y_test, pred_test, zero_division=0), 4))
    print("AUC-ROC       :", round(roc_auc_score(y_test, probabilites_test), 4))

    print("\nMatrice de confusion :")
    print(matrice)

    # 7. Sauvegarde des résultats
    dossier = BASE_DIR / "reports"
    dossier.mkdir(exist_ok=True)

    tableau.to_csv(
        dossier / "validation_seuils.csv",
        index=False,
    )

    pd.DataFrame(
        matrice,
        index=["Réel à temps", "Réel en retard"],
        columns=["Prédit à temps", "Prédit en retard"],
    ).to_csv(dossier / "matrice_test_finale.csv")

    print("\nRésultats sauvegardés dans reports/")


if __name__ == "__main__":
    main()
