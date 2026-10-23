"""Entraîne le modèle de prédiction des retards en suivant le même processus que
le notebook Analyse.ipynb, et enregistre l'expérience dans MLflow (DagsHub).

Étapes (identiques au notebook) :
    1. Chargement + conversion des dates
    2. Ingénierie des variables (delaiPrevuJours, jourSemaine, mois)
    3. EDA (graphiques enregistrés comme artefacts MLflow)
    4. Sélection des variables + encodage LabelEncoder
    5. Split 80/20 stratifié
    6. RandomForest (300 arbres, profondeur 12, min_samples_leaf 5, class_weight balanced)
    7. Évaluation : accuracy, classification_report, matrice de confusion, AUC-ROC, courbe ROC
    8. Importance des variables
    9. Sauvegarde dans models/ : model_retard.pkl, encoders.pkl, features.pkl

Authentification DagsHub (variables d'environnement) :
    MLFLOW_TRACKING_USERNAME   votre nom d'utilisateur DagsHub
    MLFLOW_TRACKING_PASSWORD   votre token DagsHub (Settings > Tokens)

Optionnel :
    DAGSHUB_REPO_OWNER / DAGSHUB_REPO_NAME   (défaut : OmegaRak / projet-IA)
    MLFLOW_TRACKING_URI                      (remplace l'URI DagsHub, ex. sqlite:///mlflow.db pour un test local)

Exemples :
    python src/train_mlflow.py
    python src/train_mlflow.py --run-name rf-profondeur-8 --max-depth 8
"""
import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'interface graphique nécessaire
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES = [
    "transporteur", "modeExpedition", "region", "categorieProduit",
    "quantite", "distanceKm", "delaiPrevuJours", "jourSemaine", "mois",
]
COLONNES_CATEGORIELLES = ["transporteur", "modeExpedition", "region", "categorieProduit", "jourSemaine"]
CIBLE = "retard"
NOMS_CLASSES = ["A temps", "En retard"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=BASE_DIR / "data" / "Livraisons.csv")
    p.add_argument("--models-dir", type=Path, default=BASE_DIR / "models",
                   help="Dossier de sauvegarde des .pkl (comme ../models dans le notebook)")
    p.add_argument("--experiment", default="retard-livraison", help="Experience")
    p.add_argument("--run-name", default=None, help="Nom du run (optionnel)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    # Hyperparamètres (valeurs par défaut = celles du notebook)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Tracking DagsHub (variables d'environnement)
# --------------------------------------------------------------------------- #
def configurer_tracking() -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        owner = os.getenv("DAGSHUB_REPO_OWNER", "OmegaRak")
        repo = os.getenv("DAGSHUB_REPO_NAME", "projet-IA")
        uri = f"https://dagshub.com/{owner}/{repo}.mlflow"
    if "dagshub.com" in uri and not (
        os.getenv("MLFLOW_TRACKING_USERNAME") and os.getenv("MLFLOW_TRACKING_PASSWORD")
    ):
        sys.exit(
            "Identifiants DagsHub manquants. Définissez MLFLOW_TRACKING_USERNAME et "
            "MLFLOW_TRACKING_PASSWORD (token DagsHub) avant de lancer le script."
        )
    mlflow.set_tracking_uri(uri)
    return uri


# --------------------------------------------------------------------------- #
# Données
# --------------------------------------------------------------------------- #
def charger_donnees(chemin: Path) -> pd.DataFrame:
    df = pd.read_csv(chemin)
    print(df.shape)
    df["dateCommande"] = pd.to_datetime(df["dateCommande"])
    df["datePrevue"] = pd.to_datetime(df["datePrevue"])
    # Ingénierie des variables liées aux dates
    df["delaiPrevuJours"] = (df["datePrevue"] - df["dateCommande"]).dt.days
    df["jourSemaine"] = df["dateCommande"].dt.day_name()
    df["mois"] = df["dateCommande"].dt.month
    return df


def md5(chemin: Path) -> str:
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def sauver(fig, chemin: Path):
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# EDA (mêmes graphiques que le notebook)
# --------------------------------------------------------------------------- #
def faire_eda(df: pd.DataFrame, dossier: Path):
    dossier.mkdir(parents=True, exist_ok=True)

    print("Valeurs manquantes :\n", df.isnull().sum())
    print("\nRépartition de la cible (%) :\n", df[CIBLE].value_counts(normalize=True) * 100)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=CIBLE, ax=ax)
    ax.set_title("Répartition des retards (0=à temps, 1=en retard)")
    sauver(fig, dossier / "repartition_retards.png")

    taux_transp = df.groupby("transporteur")[CIBLE].mean().sort_values(ascending=False)
    print("\nTaux de retard par transporteur :\n", taux_transp)
    fig, ax = plt.subplots(figsize=(7, 4))
    taux_transp.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_title("Taux de retard par transporteur")
    ax.set_ylabel("Taux de retard")
    plt.setp(ax.get_xticklabels(), rotation=45)
    sauver(fig, dossier / "taux_par_transporteur.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="modeExpedition", y=CIBLE, estimator=np.mean, ax=ax)
    ax.set_title("Taux de retard par mode d'expédition")
    ax.set_ylabel("Taux de retard moyen")
    sauver(fig, dossier / "taux_par_mode_expedition.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x=CIBLE, y="distanceKm", ax=ax)
    ax.set_title("Distance selon le statut de livraison")
    ax.set_xlabel("Retard (0 = non, 1 = oui)")
    sauver(fig, dossier / "distance_selon_retard.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df, x="mois", y=CIBLE, estimator=np.mean, color="coral", ax=ax)
    ax.set_title("Taux de retard par mois")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Taux de retard moyen")
    sauver(fig, dossier / "taux_par_mois.png")

    taux_region = df.groupby("region")[CIBLE].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    taux_region.plot(kind="bar", color="darkorange", ax=ax)
    ax.set_title("Taux de retard par région (Madagascar)")
    ax.set_ylabel("Taux de retard moyen")
    plt.setp(ax.get_xticklabels(), rotation=45)
    sauver(fig, dossier / "taux_par_region.png")

    num_cols = ["quantite", "distanceKm", "delaiPrevuJours", "mois", CIBLE]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Corrélations entre variables numériques")
    sauver(fig, dossier / "correlations.png")


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    uri = configurer_tracking()

    # --- 1-2. Données et ingénierie des variables ---
    df = charger_donnees(args.data)

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # --- 3. EDA ---
        faire_eda(df, tmp / "eda")
        mlflow.log_artifacts(str(tmp / "eda"), artifact_path="eda")

        # --- 4. Variables explicatives / cible + encodage (comme dans la section 2 du notebook) ---
        X = df[FEATURES].copy()
        y = df[CIBLE].copy()

        encoders = {}  # conservés pour être réutilisés dans l'API
        for col in COLONNES_CATEGORIELLES:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

        # --- 5. Split 80/20 stratifié ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        print("Train :", X_train.shape, "| Test :", X_test.shape)
        print("Taux de retard (train) :", y_train.mean().round(3))
        print("Taux de retard (test)  :", y_test.mean().round(3))

        # --- Paramètres et informations de contexte ---
        params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "class_weight": "balanced",  # gère le déséquilibre des classes
            "random_state": args.seed,
            "n_jobs": -1,
        }
        mlflow.log_params({f"rf_{k}": v for k, v in params.items()})
        mlflow.log_params({
            "test_size": args.test_size,
            "n_lignes": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": ",".join(FEATURES),
        })
        mlflow.set_tags({
            "modele": "RandomForestClassifier",
            "dataset": args.data.name,
            "dataset_md5": md5(args.data),
            "sklearn_version": sklearn.__version__,
        })

        # --- 6. Entraînement ---
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # --- 7. Évaluation ---
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        print("Prédictions     :", y_pred[:10])
        print("Valeurs réelles :", y_test.iloc[:10].values)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        rapport_txt = classification_report(y_test, y_pred, target_names=NOMS_CLASSES)
        rapport = classification_report(y_test, y_pred, target_names=NOMS_CLASSES, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        print("\n=== Rapport de classification ===")
        print(rapport_txt)
        print("Matrice de confusion :\n", cm)
        print("Accuracy :", round(acc, 3), "| AUC-ROC :", round(auc, 3))

        mlflow.log_metrics({
            "test_accuracy": acc,
            "test_roc_auc": auc,
            "train_roc_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
            "test_precision_retard": rapport["En retard"]["precision"],
            "test_recall_retard": rapport["En retard"]["recall"],
            "test_f1_retard": rapport["En retard"]["f1-score"],
            "test_precision_a_temps": rapport["A temps"]["precision"],
            "test_recall_a_temps": rapport["A temps"]["recall"],
            "test_f1_a_temps": rapport["A temps"]["f1-score"],
        })

        evaluation = tmp / "evaluation"
        evaluation.mkdir()
        (evaluation / "classification_report.txt").write_text(rapport_txt, encoding="utf-8")

        # Matrice de confusion
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=NOMS_CLASSES, yticklabels=NOMS_CLASSES, ax=ax)
        ax.set_xlabel("Prédiction")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion")
        sauver(fig, evaluation / "confusion_matrix.png")

        # Courbe ROC
        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title("Courbe ROC")
        sauver(fig, evaluation / "roc_curve.png")

        # --- 8. Importance des variables ---
        importances = pd.DataFrame({
            "Variable": X_train.columns,
            "Importance": model.feature_importances_,
        }).sort_values(by="Importance", ascending=False)
        print("\nImportance des variables :\n", importances)
        importances.to_csv(evaluation / "feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        importances.set_index("Variable")["Importance"].plot(kind="barh", color="teal", ax=ax)
        ax.invert_yaxis()
        ax.set_title("Importance des variables")
        sauver(fig, evaluation / "feature_importance.png")

        mlflow.log_artifacts(str(evaluation), artifact_path="evaluation")

        # --- 9. Sauvegarde locale dans models/ (comme le notebook) ---
        args.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.models_dir / "model_retard.pkl")
        joblib.dump(encoders, args.models_dir / "encoders.pkl")
        joblib.dump(list(X_train.columns), args.models_dir / "features.pkl")
        print(f"Modèle, encoders et features sauvegardés dans {args.models_dir}")
        # Envoi dans MLflow sous leurs noms d'origine (le dossier local disparaît en CI)
        for nom in ("model_retard.pkl", "encoders.pkl", "features.pkl"):
            mlflow.log_artifact(str(args.models_dir / nom), artifact_path="models")

        # --- Modèle + préprocessing dans MLflow ---
        # save_model + log_artifacts (plutôt que log_model) : compatible avec le serveur MLflow de DagsHub.
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.save_model(
            model,
            str(tmp / "model"),
            signature=signature,
            input_example=X_train.head(5),
            pip_requirements=[f"scikit-learn=={sklearn.__version__}"],
            # Même format pickle que les .pkl du projet (le format skops par défaut
            # de MLflow refuse les arbres scikit-learn comme "types non fiables").
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        mlflow.log_artifacts(str(tmp / "model"), artifact_path="model")

        (tmp / "preprocessing").mkdir()
        joblib.dump(encoders, tmp / "preprocessing" / "encoders.pkl")
        joblib.dump(list(X_train.columns), tmp / "preprocessing" / "features.pkl")
        mlflow.log_artifacts(str(tmp / "preprocessing"), artifact_path="preprocessing")

        print(f"\nRun terminé : {run.info.run_id}")
        print(f"Tracking URI : {uri}")
        print(f"AUC test = {auc:.3f} | accuracy test = {acc:.3f}")
        if "dagshub.com" in uri:
            print(f"Voir les expériences : {uri.removesuffix('.mlflow')}/experiments")


if __name__ == "__main__":
    main()