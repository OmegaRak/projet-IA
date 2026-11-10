
"""Optimisation du Gradient Boosting avec GridSearchCV et MLflow."""

import mlflow
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
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
    configurer_tracking,
    md5,
)


def main():

    # 1. Configuration MLflow
    uri = configurer_tracking()
    mlflow.set_experiment("retard-livraison")

    print("Tracking URI :", uri)

    # 2. Chargement des données
    chemin = BASE_DIR / "data" / "Livraisons.csv"
    df = charger_donnees(chemin)

    X = df[FEATURES].copy()
    y = df[CIBLE].copy()

    # 3. Encodage
    for col in COLONNES_CATEGORIELLES:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # 4. Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTrain :", X_train.shape)
    print("Test :", X_test.shape)

    # 5. Gestion du déséquilibre des classes
    poids = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    # 6. Définition des paramètres à tester
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3],
        "min_samples_leaf": [5, 10],
    }

    # 7. Validation croisée stratifiée
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    modele = GradientBoostingClassifier(
        random_state=42
    )

    recherche = GridSearchCV(
        estimator=modele,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    # 8. Optimisation
    print("\nDébut de GridSearchCV...")

    recherche.fit(
        X_train,
        y_train,
        sample_weight=poids,
    )

    meilleur_modele = recherche.best_estimator_

    print("\nMeilleurs paramètres :")
    print(recherche.best_params_)

    print(
        "\nMeilleur AUC-ROC en Cross-Validation :",
        round(recherche.best_score_, 4),
    )

    # 9. Évaluation finale sur le jeu de test
    y_pred = meilleur_modele.predict(X_test)
    y_proba = meilleur_modele.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print("\n========== RÉSULTATS FINAUX ==========")
    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1-score  :", round(f1, 4))
    print("AUC-ROC   :", round(auc, 4))

    # 10. Enregistrement dans MLflow
    with mlflow.start_run(run_name="Gradient-Boosting-Optimise"):

        mlflow.log_params(recherche.best_params_)

        mlflow.log_param("algorithme", "GradientBoostingClassifier")
        mlflow.log_param("methode_optimisation", "GridSearchCV")
        mlflow.log_param("cross_validation", 5)
        mlflow.log_param("scoring", "roc_auc")
        mlflow.log_param("nombre_combinaisons", len(recherche.cv_results_["params"]))

        mlflow.log_metrics({
            "cv_best_roc_auc": recherche.best_score_,
            "test_accuracy": accuracy,
            "test_precision_retard": precision,
            "test_recall_retard": recall,
            "test_f1_retard": f1,
            "test_roc_auc": auc,
        })

        mlflow.set_tags({
            "dataset": chemin.name,
            "dataset_md5": md5(chemin),
            "type": "optimisation_hyperparametres",
        })

    # 11. Sauvegarde des résultats
    dossier = BASE_DIR / "reports"
    dossier.mkdir(exist_ok=True)

    tableau = pd.DataFrame(recherche.cv_results_)
    tableau = tableau.sort_values(
        by="mean_test_score",
        ascending=False,
    )

    tableau.to_csv(
        dossier / "gridsearch_gradient_boosting.csv",
        index=False,
    )

    print("\nRésultats GridSearch sauvegardés.")
    print("\nTop 5 des combinaisons :")
    print(
        tableau[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].head(5).to_string(index=False)
    )

    print("\nExpérience enregistrée dans MLflow :", uri)


if __name__ == "__main__":
    main()
