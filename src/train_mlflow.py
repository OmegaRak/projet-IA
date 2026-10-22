"""Entraîne le modèle de prédiction des retards (mêmes réglages que le notebook)
et enregistre l'expérience dans MLflow (hébergé sur DagsHub).

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
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURES = [
    "transporteur", "modeExpedition", "region", "categorieProduit",
    "quantite", "distanceKm", "delaiPrevuJours", "jourSemaine", "mois",
]
CATEGORIELLES = ["transporteur", "modeExpedition", "region", "categorieProduit", "jourSemaine"]
CIBLE = "retard"
SEUIL = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=BASE_DIR / "data" / "Livraisons.csv")
    p.add_argument("--experiment", default="retard-livraison", help="Experience")
    p.add_argument("--run-name", default=None, help="Nom du run (optionnel)")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    # Hyperparamètres (valeurs par défaut = celles du notebook)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    return p.parse_args()


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


def charger_donnees(chemin: Path) -> pd.DataFrame:
    df = pd.read_csv(chemin, parse_dates=["dateCommande", "datePrevue"])
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


def main():
    args = parse_args()
    uri = configurer_tracking()

    # --- Données et prétraitement (identique au notebook) ---
    df = charger_donnees(args.data)
    X = df[FEATURES].copy()
    y = df[CIBLE]

    encoders = {}
    for col in CATEGORIELLES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "class_weight": "balanced",
        "random_state": args.seed,
        "n_jobs": -1,
    }

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # --- Paramètres et informations de contexte ---
        mlflow.log_params({f"rf_{k}": v for k, v in params.items()})
        mlflow.log_params({
            "test_size": args.test_size,
            "seuil_decision": SEUIL,
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

        # --- Entraînement ---
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        # --- Métriques (jeu de test) + baseline "toujours à l'heure" ---
        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test = (proba_test >= SEUIL).astype(int)
        baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

        mlflow.log_metrics({
            "test_accuracy": accuracy_score(y_test, pred_test),
            "test_precision_retard": precision_score(y_test, pred_test),
            "test_recall_retard": recall_score(y_test, pred_test),
            "test_f1_retard": f1_score(y_test, pred_test),
            "test_roc_auc": roc_auc_score(y_test, proba_test),
            "train_roc_auc": roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
            "baseline_accuracy": accuracy_score(y_test, baseline.predict(X_test)),
        })

        # --- Graphiques ---
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred_test, display_labels=["A temps", "En retard"], cmap="Blues", ax=ax
        )
        ax.set_title("Matrice de confusion (test)")
        fig.savefig(tmp / "confusion_matrix.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 4))
        RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax)
        ax.set_title("Courbe ROC (test)")
        fig.savefig(tmp / "roc_curve.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        # --- Importance des variables : par impureté ET par permutation ---
        perm = permutation_importance(
            model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=args.seed, n_jobs=-1
        )
        importances = pd.DataFrame({
            "feature": FEATURES,
            "importance_impurete": model.feature_importances_,
            "importance_permutation_auc": perm.importances_mean,
        }).sort_values("importance_permutation_auc", ascending=False)
        importances.to_csv(tmp / "feature_importance.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        importances.set_index("feature")["importance_permutation_auc"].sort_values().plot.barh(ax=ax)
        ax.set_title("Importance par permutation (baisse d'AUC)")
        fig.savefig(tmp / "permutation_importance.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

        mlflow.log_artifacts(str(tmp), artifact_path="evaluation")
        for f in tmp.iterdir():
            f.unlink()

        # --- Modèle + préprocessing ---
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
        joblib.dump(FEATURES, tmp / "preprocessing" / "features.pkl")
        mlflow.log_artifacts(str(tmp / "preprocessing"), artifact_path="preprocessing")

        print(f"\nRun terminé : {run.info.run_id}")
        print(f"Tracking URI : {uri}")
        print(f"AUC test = {roc_auc_score(y_test, proba_test):.3f} | "
              f"accuracy test = {accuracy_score(y_test, pred_test):.3f} | "
              f"baseline = {accuracy_score(y_test, baseline.predict(X_test)):.3f}")
        if "dagshub.com" in uri:
            print(f"Voir les expériences : {uri.removesuffix('.mlflow')}/experiments")


if __name__ == "__main__":
    main()
