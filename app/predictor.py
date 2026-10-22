"""Chargement du modèle et logique de prédiction (indépendante de FastAPI)."""
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from .schemas import (
    CategorieProduit,
    Livraison,
    ModeExpedition,
    Region,
    Transporteur,
)

# Noms anglais, car ce sont ceux appris par l'encodeur (dt.day_name() à l'entraînement).
# Indexés par date.weekday() (lundi = 0) : indépendant de la locale du serveur.
JOURS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_MODALITES_ATTENDUES = {
    "transporteur": Transporteur,
    "modeExpedition": ModeExpedition,
    "region": Region,
    "categorieProduit": CategorieProduit,
}


def verifier_modalites(encoders: dict) -> None:
    """Échoue au démarrage si les Enum de l'API divergent des encodeurs du modèle."""
    for col, enum_cls in _MODALITES_ATTENDUES.items():
        attendu = {e.value for e in enum_cls}
        appris = set(encoders[col].classes_)
        if attendu != appris:
            raise RuntimeError(
                f"Modalités incohérentes pour '{col}': "
                f"seulement dans l'API={sorted(attendu - appris)}, "
                f"seulement dans le modèle={sorted(appris - attendu)}"
            )


class Predictor:
    def __init__(self, model_dir: Path, seuil: float = 0.5):
        self.model = joblib.load(model_dir / "model_retard.pkl")
        self.encoders = joblib.load(model_dir / "encoders.pkl")
        self.features: list[str] = joblib.load(model_dir / "features.pkl")
        self.seuil = seuil

        verifier_modalites(self.encoders)

        # Le modèle a été sauvegardé avec n_jobs=-1 : pour de petites requêtes,
        # le coût de parallélisation dépasse le gain.
        self.model.set_params(n_jobs=1)
        self._idx_retard = list(self.model.classes_).index(1)

    def _vers_dataframe(self, livraisons: Sequence[Livraison]) -> pd.DataFrame:
        lignes = [
            {
                "transporteur": l.transporteur.value,
                "modeExpedition": l.modeExpedition.value,
                "region": l.region.value,
                "categorieProduit": l.categorieProduit.value,
                "quantite": l.quantite,
                "distanceKm": l.distanceKm,
                "delaiPrevuJours": (l.datePrevue - l.dateCommande).days,
                "jourSemaine": JOURS[l.dateCommande.weekday()],
                "mois": l.dateCommande.month,
            }
            for l in livraisons
        ]
        # Même ordre de colonnes qu'à l'entraînement
        df = pd.DataFrame(lignes)[self.features]
        for col, enc in self.encoders.items():
            df[col] = enc.transform(df[col])
        return df

    def probabilites(self, livraisons: Sequence[Livraison]) -> list[float]:
        X = self._vers_dataframe(livraisons)
        return self.model.predict_proba(X)[:, self._idx_retard].tolist()

    def predire(self, livraisons: Sequence[Livraison]) -> list[dict]:
        return [
            {
                "retard": p >= self.seuil,
                "probabiliteRetard": round(p, 4),
                "seuil": self.seuil,
            }
            for p in self.probabilites(livraisons)
        ]

    def info(self) -> dict:
        return {
            "typeModele": type(self.model).__name__,
            "nombreArbres": int(self.model.n_estimators),
            "features": self.features,
            "modalites": {
                col: [str(c) for c in self.encoders[col].classes_]
                for col in _MODALITES_ATTENDUES
            },
            "seuil": self.seuil,
        }
