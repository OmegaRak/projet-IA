"""API FastAPI de prédiction des retards de livraison."""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .predictor import Predictor
from .schemas import (
    InfoModele,
    Livraison,
    LotLivraisons,
    LotPredictions,
    Prediction,
    Sante,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Le modèle est chargé une seule fois au démarrage
    app.state.predictor = Predictor(config.MODEL_DIR, config.SEUIL_RETARD)
    yield


app = FastAPI(
    title="API de prédiction des retards de livraison",
    description="Prédit si une livraison (Madagascar) risque d'arriver en retard.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


# Endpoints déclarés avec `def` (et non `async def`) : le calcul du modèle est
# synchrone/CPU, FastAPI les exécute dans un thread sans bloquer la boucle async.


@app.get("/health", response_model=Sante, tags=["Technique"])
def health(predictor: Predictor = Depends(get_predictor)):
    return {"status": "ok", "modeleCharge": predictor.model is not None}


@app.get("/model/info", response_model=InfoModele, tags=["Technique"])
def model_info(predictor: Predictor = Depends(get_predictor)):
    return predictor.info()


@app.post("/predict", response_model=Prediction, tags=["Prédiction"])
def predict(livraison: Livraison, predictor: Predictor = Depends(get_predictor)):
    return predictor.predire([livraison])[0]


@app.post("/predict/batch", response_model=LotPredictions, tags=["Prédiction"])
def predict_batch(lot: LotLivraisons, predictor: Predictor = Depends(get_predictor)):
    return {"predictions": predictor.predire(lot.livraisons)}
