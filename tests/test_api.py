import warnings
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app import config
from app.main import app

EXEMPLE = {
    "dateCommande": "2025-03-11",
    "datePrevue": "2025-03-17",
    "transporteur": "Trans Mahefa",
    "modeExpedition": "Standard",
    "region": "Toamasina",
    "categorieProduit": "Artisanat",
    "quantite": 9,
    "distanceKm": 361.9,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # déclenche le lifespan (chargement du modèle)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "modeleCharge": True}


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["typeModele"] == "RandomForestClassifier"
    assert len(body["features"]) == 9
    assert "Trans Mahefa" in body["modalites"]["transporteur"]


def test_predict_ok(client):
    r = client.post("/predict", json=EXEMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probabiliteRetard"] <= 1
    assert body["retard"] == (body["probabiliteRetard"] >= body["seuil"])


@pytest.mark.parametrize(
    "champ,valeur",
    [
        ("transporteur", "Inconnu Express"),
        ("region", "Paris"),
        ("modeExpedition", "Teleportation"),
        ("categorieProduit", "Licornes"),
        ("quantite", 0),
        ("distanceKm", -5),
        ("dateCommande", "pas-une-date"),
    ],
)
def test_predict_valeurs_invalides(client, champ, valeur):
    r = client.post("/predict", json={**EXEMPLE, champ: valeur})
    assert r.status_code == 422


def test_predict_champ_manquant(client):
    payload = {k: v for k, v in EXEMPLE.items() if k != "region"}
    assert client.post("/predict", json=payload).status_code == 422


def test_date_prevue_avant_commande(client):
    payload = {**EXEMPLE, "datePrevue": "2025-03-01"}
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_coherent_avec_unitaire(client):
    autre = {**EXEMPLE, "transporteur": "Service Rapide", "region": "Toliara",
             "distanceKm": 940.0, "dateCommande": "2025-01-10", "datePrevue": "2025-01-20"}
    batch = client.post("/predict/batch", json={"livraisons": [EXEMPLE, autre]})
    assert batch.status_code == 200
    preds = batch.json()["predictions"]
    assert len(preds) == 2
    for payload, pred in zip([EXEMPLE, autre], preds):
        assert client.post("/predict", json=payload).json() == pred


def test_batch_vide_ou_trop_grand(client):
    assert client.post("/predict/batch", json={"livraisons": []}).status_code == 422
    trop = {"livraisons": [EXEMPLE] * (config.MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=trop).status_code == 422


def test_fidelite_avec_le_notebook(client):
    """L'API doit donner exactement les mêmes probabilités que le pipeline du notebook
    (encodage des colonnes puis predict_proba) sur de vraies lignes du dataset."""
    csv = Path(config.BASE_DIR / "data" / "Livraisons.csv")
    df = pd.read_csv(csv, parse_dates=["dateCommande", "datePrevue"]).head(25)

    model = joblib.load(config.MODEL_DIR / "model_retard.pkl")
    encoders = joblib.load(config.MODEL_DIR / "encoders.pkl")
    features = joblib.load(config.MODEL_DIR / "features.pkl")

    X = df.copy()
    X["delaiPrevuJours"] = (df["datePrevue"] - df["dateCommande"]).dt.days
    X["jourSemaine"] = df["dateCommande"].dt.day_name()
    X["mois"] = df["dateCommande"].dt.month
    X = X[features].copy()
    for col, enc in encoders.items():
        X[col] = enc.transform(X[col])
    attendu = model.predict_proba(X)[:, 1]

    livraisons = [
        {
            "dateCommande": r.dateCommande.strftime("%Y-%m-%d"),
            "datePrevue": r.datePrevue.strftime("%Y-%m-%d"),
            "transporteur": r.transporteur,
            "modeExpedition": r.modeExpedition,
            "region": r.region,
            "categorieProduit": r.categorieProduit,
            "quantite": int(r.quantite),
            "distanceKm": float(r.distanceKm),
        }
        for r in df.itertuples()
    ]
    r = client.post("/predict/batch", json={"livraisons": livraisons})
    assert r.status_code == 200
    obtenu = [p["probabiliteRetard"] for p in r.json()["predictions"]]
    for a, o in zip(attendu, obtenu):
        assert abs(a - o) < 1e-4  # l'API arrondit à 4 décimales
