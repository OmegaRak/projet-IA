
"""Évaluation détaillée du Gradient Boosting optimisé."""

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
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

    # 2. Encodage des variables catégorielles
    for col in COLONNES_CATEGORIELLES:
        encoder = LabelEncoder()
        X[col] = encoder.fit_transform(X[col])

    # 3. Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4. Paramètres optimisés par GridSearchCV
    modele = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        min_samples_leaf=10,
        random_state=42,
    )

    # 5. Entraînement
    poids = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    modele.fit(
        X_train,
        y_train,
        sample_weight=poids,
    )

    # 6. Prédictions
    y_pred = modele.predict(X_test)
    y_proba = modele.predict_proba(X_test)[:, 1]

    # 7. Matrice de confusion
    matrice = confusion_matrix(y_test, y_pred)

    print("\n========== MATRICE DE CONFUSION ==========")
    print("                 Prédit à temps | Prédit en retard")
    print(f"Réel à temps          {matrice[0, 0]:5d}       | {matrice[0, 1]:5d}")
    print(f"Réel en retard        {matrice[1, 0]:5d}       | {matrice[1, 1]:5d}")

    # 8. Rapport de classification
    print("\n========== RAPPORT DE CLASSIFICATION ==========")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["À temps", "En retard"],
            zero_division=0,
        )
    )

    # 9. Métriques globales
    print("========== MÉTRIQUES GLOBALES ==========")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("AUC-ROC  :", round(roc_auc_score(y_test, y_proba), 4))

    # 10. Sauvegarde de la matrice
    dossier = BASE_DIR / "reports"
    dossier.mkdir(exist_ok=True)

    tableau = pd.DataFrame(
        matrice,
        index=["Réel à temps", "Réel en retard"],
        columns=["Prédit à temps", "Prédit en retard"],
    )

    tableau.to_csv(dossier / "matrice_confusion.csv")

    print("\nMatrice de confusion sauvegardée dans reports/")


if __name__ == "__main__":
    main()
