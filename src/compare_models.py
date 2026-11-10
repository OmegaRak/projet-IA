
"""Compare plusieurs algorithmes de classification avec MLflow."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Réutilisation du code déjà existant
from train_mlflow import (
    BASE_DIR,
    FEATURES,
    CIBLE,
    COLONNES_CATEGORIELLES,
    charger_donnees,
    configurer_tracking,
    md5,
)


def main():
    # 1. Configuration MLflow
    uri = configurer_tracking()
    experiment_name = "retard-livraison"
    mlflow.set_experiment(experiment_name)

    print("Tracking URI :", uri)

    # 2. Chargement et préparation des données
    chemin = BASE_DIR / "data" / "Livraisons.csv"
    df = charger_donnees(chemin)

    X = df[FEATURES].copy()
    y = df[CIBLE].copy()

    # Encodage des variables catégorielles
    for col in COLONNES_CATEGORIELLES:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # 3. Même séparation train/test que le modèle précédent
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain :", X_train.shape)
    print("Test  :", X_test.shape)

    # 4. Définition des modèles
    modeles = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
        ),
    }

    resultats = []

    # 5. Entraînement et évaluation
    for nom, modele in modeles.items():

        print(f"\n{'=' * 50}")
        print("Entraînement :", nom)
        print("=" * 50)

        with mlflow.start_run(run_name=nom):

            # Gestion du déséquilibre des classes
            if nom == "Gradient Boosting":
                poids = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
                modele.fit(X_train, y_train, sample_weight=poids)
            else:
                modele.fit(X_train, y_train)

            # Prédictions
            y_pred = modele.predict(X_test)
            y_proba = modele.predict_proba(X_test)[:, 1]

            # Métriques
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)

            # Enregistrement dans MLflow
            mlflow.log_params({
                "algorithme": nom,
                "test_size": 0.2,
                "random_state": 42,
                "n_features": len(FEATURES),
                "n_train": len(X_train),
                "n_test": len(X_test),
            })

            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision_retard": precision,
                "recall_retard": recall,
                "f1_retard": f1,
                "roc_auc": auc,
            })

            mlflow.set_tags({
                "dataset": chemin.name,
                "dataset_md5": md5(chemin),
                "type": "comparaison_modeles",
            })

            resultats.append({
                "Modele": nom,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-score": f1,
                "AUC-ROC": auc,
            })

            print(f"Accuracy  : {accuracy:.4f}")
            print(f"Precision : {precision:.4f}")
            print(f"Recall    : {recall:.4f}")
            print(f"F1-score  : {f1:.4f}")
            print(f"AUC-ROC   : {auc:.4f}")

    # 6. Tableau comparatif
    tableau = pd.DataFrame(resultats)
    tableau = tableau.sort_values(
        by="AUC-ROC",
        ascending=False,
    )

    print("\n\n========== COMPARAISON FINALE ==========")
    print(tableau.to_string(index=False))

    # 7. Sauvegarde locale des résultats
    dossier = BASE_DIR / "reports"
    dossier.mkdir(exist_ok=True)

    fichier = dossier / "comparaison_modeles.csv"
    tableau.to_csv(fichier, index=False)

    print("\nRésultats sauvegardés :", fichier)

    # 8. Meilleur modèle selon AUC-ROC
    meilleur = tableau.iloc[0]

    print("\n========== MEILLEUR MODÈLE ==========")
    print("Algorithme :", meilleur["Modele"])
    print("AUC-ROC    :", round(meilleur["AUC-ROC"], 4))
    print("Accuracy   :", round(meilleur["Accuracy"], 4))

    print("\nToutes les expériences sont enregistrées dans MLflow.")
    print("Consultez :", uri)


if __name__ == "__main__":
    main()
