
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
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "modeleCharge": True,
    }


def test_model_info(client):
    r = client.get("/model/info")

    assert r.status_code == 200

    body = r.json()

    assert body["typeModele"] == "GradientBoostingClassifier"
    assert body["nombreArbres"] == 200
    assert len(body["features"]) == 9
    assert body["seuil"] == 0.4
    assert "Trans Mahefa" in body["modalites"]["transporteur"]


def test_predict_ok(client):
    r = client.post("/predict", json=EXEMPLE)

    assert r.status_code == 200

    body = r.json()

    assert 0 <= body["probabiliteRetard"] <= 1
    assert body["seuil"] == 0.4
    assert body["retard"] == (
        body["probabiliteRetard"] >= body["seuil"]
    )


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
    r = client.post(
        "/predict",
        json={**EXEMPLE, champ: valeur},
    )

    assert r.status_code == 422


def test_predict_champ_manquant(client):
    payload = {
        k: v
        for k, v in EXEMPLE.items()
        if k != "region"
    }

    assert client.post("/predict", json=payload).status_code == 422


def test_date_prevue_avant_commande(client):
    payload = {
        **EXEMPLE,
        "datePrevue": "2025-03-01",
    }

    assert client.post("/predict", json=payload).status_code == 422


def test_batch_coherent_avec_unitaire(client):
    autre = {
        **EXEMPLE,
        "transporteur": "Service Rapide",
        "region": "Toliara",
        "distanceKm": 940.0,
        "dateCommande": "2025-01-10",
        "datePrevue": "2025-01-20",
    }

    livraisons = [EXEMPLE, autre]

    batch = client.post(
        "/predict/batch",
        json={"livraisons": livraisons},
    )

    assert batch.status_code == 200

    predictions = batch.json()["predictions"]

    assert len(predictions) == 2

    for payload, prediction in zip(livraisons, predictions):
        resultat = client.post(
            "/predict",
            json=payload,
        )

        assert resultat.status_code == 200
        assert resultat.json() == prediction


def test_batch_vide_ou_trop_grand(client):
    assert client.post(
        "/predict/batch",
        json={"livraisons": []},
    ).status_code == 422

    trop = {
        "livraisons": [EXEMPLE] * (config.MAX_BATCH + 1)
    }

    assert client.post(
        "/predict/batch",
        json=trop,
    ).status_code == 422


def test_fidelite_avec_le_pipeline_final(client):
    """L'API doit utiliser exactement le même pipeline final."""

    csv = config.BASE_DIR / "data" / "Livraisons.csv"

    df = pd.read_csv(
        csv,
        parse_dates=["dateCommande", "datePrevue"],
    ).head(25)

    # Chargement du pipeline final
    modele = joblib.load(
        config.MODEL_DIR / "gradient_boosting_final.pkl"
    )

    features = joblib.load(
        config.MODEL_DIR / "features_final.pkl"
    )

    seuil = joblib.load(
        config.MODEL_DIR / "seuil_decision.pkl"
    )

    # Préparation des données comme dans l'entraînement
    X = df.copy()

    X["delaiPrevuJours"] = (
        df["datePrevue"] - df["dateCommande"]
    ).dt.days

    X["jourSemaine"] = df["dateCommande"].dt.day_name()

    X["mois"] = df["dateCommande"].dt.month

    X = X[features]

    # Le pipeline effectue lui-même le prétraitement
    probabilites = modele.predict_proba(X)[:, 1]

    predictions_attendues = (
        probabilites >= seuil
    )

    # Préparation des requêtes API
    livraisons = [
        {
            "dateCommande": ligne.dateCommande.strftime("%Y-%m-%d"),
            "datePrevue": ligne.datePrevue.strftime("%Y-%m-%d"),
            "transporteur": ligne.transporteur,
            "modeExpedition": ligne.modeExpedition,
            "region": ligne.region,
            "categorieProduit": ligne.categorieProduit,
            "quantite": int(ligne.quantite),
            "distanceKm": float(ligne.distanceKm),
        }
        for ligne in df.itertuples()
    ]

    reponse = client.post(
        "/predict/batch",
        json={"livraisons": livraisons},
    )

    assert reponse.status_code == 200

    predictions_api = reponse.json()["predictions"]

    assert len(predictions_api) == len(probabilites)

    for probabilite, prediction_attendue, obtenu in zip(
        probabilites,
        predictions_attendues,
        predictions_api,
    ):
        assert abs(
            probabilite - obtenu["probabiliteRetard"]
        ) < 1e-4

        assert obtenu["retard"] == bool(prediction_attendue)
        assert obtenu["seuil"] == seuil
