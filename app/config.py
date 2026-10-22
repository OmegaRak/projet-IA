"""Configuration de l'API (surchargeable par variables d'environnement)."""
import os
from pathlib import Path

# api/app/config.py -> racine du dépôt = parents[2]
BASE_DIR = Path(__file__).resolve().parent.parent

# Dossier contenant model_retard.pkl, encoders.pkl, features.pkl
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))

# Seuil de décision : une livraison est "en retard" si proba >= seuil
SEUIL_RETARD = float(os.getenv("SEUIL_RETARD", "0.5"))

# Nombre maximum de livraisons par requête /predict/batch
MAX_BATCH = int(os.getenv("MAX_BATCH", "1000"))

# Origines autorisées (CORS), séparées par des virgules
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
    ).split(",")
    if o.strip()
]
