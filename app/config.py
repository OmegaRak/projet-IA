
"""Configuration de l'API."""

import os
from pathlib import Path

# Racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossier contenant le modèle final
MODEL_DIR = Path(
    os.getenv("MODEL_DIR", str(BASE_DIR / "models" / "final"))
)

# Seuil de décision par défaut
SEUIL_RETARD = float(os.getenv("SEUIL_RETARD", "0.4"))

# Nombre maximum de livraisons par requête
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))

# Origines autorisées (CORS)
CORS_ORIGINS = [
    origine.strip()
    for origine in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    ).split(",")
    if origine.strip()
]
