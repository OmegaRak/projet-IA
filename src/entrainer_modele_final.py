
"""Entraînement et sauvegarde du modèle final."""

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

    # 2. Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. Prétraitement des variables catégorielles
    preprocesseur = ColumnTransformer(
        transformers=[
            (
                "categoriel",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                COLONNES_CATEGORIELLES,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # 4. Modèle optimisé
    classifieur = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=10,
        random_state=42,
    )

    # 5. Pipeline complet
    modele_final = Pipeline([
        ("preprocesseur", preprocesseur),
        ("classifieur", classifieur),
    ])

    # 6. Entraînement
    poids = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    modele_final.fit(
        X_train,
        y_train,
        classifieur__sample_weight=poids,
    )

    # 7. Évaluation
    probabilites = modele_final.predict_proba(X_test)[:, 1]

    seuil = 0.4
    predictions = (probabilites >= seuil).astype(int)

    print("\n========== ÉVALUATION DU MODÈLE FINAL ==========")
    print("Seuil :", seuil)
    print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
    print("Precision :", round(precision_score(y_test, predictions, zero_division=0), 4))
    print("Recall :", round(recall_score(y_test, predictions, zero_division=0), 4))
    print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 4))
    print("AUC-ROC :", round(roc_auc_score(y_test, probabilites), 4))

    # 8. Sauvegarde dans un nouveau dossier
    dossier = BASE_DIR / "models" / "final"
    dossier.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        modele_final,
        dossier / "gradient_boosting_final.pkl",
    )

    joblib.dump(
        FEATURES,
        dossier / "features_final.pkl",
    )

    joblib.dump(
        seuil,
        dossier / "seuil_decision.pkl",
    )

    print("\nModèle final sauvegardé dans :", dossier)
    print("Fichiers créés avec succès !")


if __name__ == "__main__":
    main()
