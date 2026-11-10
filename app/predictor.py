
"""Chargement du modèle final et logique de prédiction."""

from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd

from .schemas import Livraison


# Noms des jours en anglais : identiques à ceux de l'entraînement
JOURS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Colonnes catégorielles utilisées par l'OrdinalEncoder
COLONNES_CATEGORIELLES = [
    "transporteur",
    "modeExpedition",
    "region",
    "categorieProduit",
    "jourSemaine",
]


class Predictor:
    def __init__(self, model_dir: Path, seuil: float = 0.4):

        # 1. Chargement du pipeline complet
        self.model = joblib.load(
            model_dir / "gradient_boosting_final.pkl"
        )

        # 2. Chargement des variables utilisées pendant l'entraînement
        self.features: list[str] = joblib.load(
            model_dir / "features_final.pkl"
        )

        # 3. Chargement du seuil sauvegardé
        self.seuil = float(
            joblib.load(model_dir / "seuil_decision.pkl")
        )

        # 4. Vérification du modèle
        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                "Le modèle chargé ne possède pas predict_proba."
            )

        if not 0 <= self.seuil <= 1:
            raise RuntimeError(
                "Le seuil de décision doit être compris entre 0 et 1."
            )

        if not hasattr(self.model, "named_steps"):
            raise RuntimeError(
                "Le modèle chargé n'est pas un Pipeline valide."
            )

        # 5. Vérification des classes
        classes = list(self.model.classes_)

        if 1 not in classes:
            raise RuntimeError(
                "La classe 1 (livraison en retard) est absente du modèle."
            )

        self._idx_retard = classes.index(1)

        # 6. Vérification du préprocesseur
        try:
            self.preprocesseur = self.model.named_steps["preprocesseur"]
            self.classifieur = self.model.named_steps["classifieur"]
            self.encodeur = (
                self.preprocesseur.named_transformers_["categoriel"]
            )
        except (KeyError, AttributeError) as erreur:
            raise RuntimeError(
                "Structure du Pipeline final invalide."
            ) from erreur

        # 7. Vérification des features
        if not self.features:
            raise RuntimeError(
                "La liste des features est vide."
            )

    def _vers_dataframe(
        self,
        livraisons: Sequence[Livraison],
    ) -> pd.DataFrame:

        """Transforme les livraisons reçues en DataFrame."""

        lignes = [
            {
                "transporteur": livraison.transporteur.value,
                "modeExpedition": livraison.modeExpedition.value,
                "region": livraison.region.value,
                "categorieProduit": livraison.categorieProduit.value,
                "quantite": livraison.quantite,
                "distanceKm": livraison.distanceKm,
                "delaiPrevuJours": (
                    livraison.datePrevue - livraison.dateCommande
                ).days,
                "jourSemaine": JOURS[
                    livraison.dateCommande.weekday()
                ],
                "mois": livraison.dateCommande.month,
            }
            for livraison in livraisons
        ]

        # Respecter exactement l'ordre des colonnes de l'entraînement
        df = pd.DataFrame(lignes)

        return df[self.features]

    def probabilites(
        self,
        livraisons: Sequence[Livraison],
    ) -> list[float]:

        """Calcule les probabilités de retard."""

        X = self._vers_dataframe(livraisons)

        # Le Pipeline réalise automatiquement l'encodage
        probabilites = self.model.predict_proba(X)

        return probabilites[:, self._idx_retard].tolist()

    def predire(
        self,
        livraisons: Sequence[Livraison],
    ) -> list[dict]:

        """Retourne les prédictions pour une ou plusieurs livraisons."""

        probabilites = self.probabilites(livraisons)

        return [
            {
                "retard": probabilite >= self.seuil,
                "probabiliteRetard": round(probabilite, 4),
                "seuil": self.seuil,
            }
            for probabilite in probabilites
        ]

    def info(self) -> dict:

        """Retourne les informations du modèle."""

        modalites = {
            colonne: [
                str(valeur)
                for valeur in categories
            ]
            for colonne, categories in zip(
                COLONNES_CATEGORIELLES,
                self.encodeur.categories_,
            )
        }

        return {
            "typeModele": type(self.classifieur).__name__,
            "nombreArbres": int(self.classifieur.n_estimators),
            "features": self.features,
            "modalites": modalites,
            "seuil": self.seuil,
        }
