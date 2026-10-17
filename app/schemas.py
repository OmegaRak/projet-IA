"""Schémas d'entrée/sortie de l'API.

Les noms de champs reprennent ceux du dataset (Livraisons.csv).
Les modalités autorisées doivent correspondre à celles vues à l'entraînement
(vérifié au démarrage, voir predictor.verifier_modalites).
"""
from datetime import date
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from .config import MAX_BATCH


class Transporteur(str, Enum):
    TRANS_MAHEFA = "Trans Mahefa"
    SONATRA = "Sonatra"
    MAMI = "Mami"
    MADA_TRANS = "Mada Trans"
    SERVICE_RAPIDE = "Service Rapide"


class ModeExpedition(str, Enum):
    STANDARD = "Standard"
    EXPRESS = "Express"
    ECONOMIQUE = "Economique"
    PRIORITAIRE = "Prioritaire"


class CategorieProduit(str, Enum):
    ELECTRONIQUE = "Electronique"
    VETEMENTS = "Vetements"
    PRODUITS_AGRICOLES = "Produits agricoles"
    ALIMENTAIRE = "Alimentaire"
    ARTISANAT = "Artisanat"
    PIECES_DETACHEES = "Pieces detachees"
    MATERIEL_MEDICAL = "Materiel medical"


class Region(str, Enum):
    ANTANANARIVO = "Antananarivo"
    ANTSIRABE = "Antsirabe"
    FIANARANTSOA = "Fianarantsoa"
    TOAMASINA = "Toamasina"
    MAHAJANGA = "Mahajanga"
    TOLIARA = "Toliara"
    ANTSIRANANA = "Antsiranana"
    MORONDAVA = "Morondava"
    AMBATONDRAZAKA = "Ambatondrazaka"
    FENERIVE_EST = "Fenerive Est"


class Livraison(BaseModel):
    dateCommande: date = Field(description="Date de la commande (YYYY-MM-DD)")
    datePrevue: date = Field(description="Date de livraison prévue (YYYY-MM-DD)")
    transporteur: Transporteur
    modeExpedition: ModeExpedition
    region: Region
    categorieProduit: CategorieProduit
    quantite: int = Field(ge=1, description="Quantité commandée (>= 1)")
    distanceKm: float = Field(ge=0, description="Distance de livraison en km")

    @model_validator(mode="after")
    def date_prevue_apres_commande(self):
        if self.datePrevue < self.dateCommande:
            raise ValueError("datePrevue doit être postérieure ou égale à dateCommande")
        return self

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dateCommande": "2025-03-11",
                    "datePrevue": "2025-03-17",
                    "transporteur": "Trans Mahefa",
                    "modeExpedition": "Standard",
                    "region": "Toamasina",
                    "categorieProduit": "Artisanat",
                    "quantite": 9,
                    "distanceKm": 361.9,
                }
            ]
        }
    }


class Prediction(BaseModel):
    retard: bool = Field(description="True si la probabilité de retard >= seuil")
    probabiliteRetard: float = Field(ge=0, le=1)
    seuil: float


class LotLivraisons(BaseModel):
    livraisons: list[Livraison] = Field(min_length=1, max_length=MAX_BATCH)


class LotPredictions(BaseModel):
    predictions: list[Prediction]


class Sante(BaseModel):
    status: str
    modeleCharge: bool


class InfoModele(BaseModel):
    typeModele: str
    nombreArbres: int
    features: list[str]
    modalites: dict[str, list[str]]
    seuil: float
