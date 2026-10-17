# API de prédiction des retards de livraison

API FastAPI qui expose le modèle Random Forest (`models/model_retard.pkl`).

## Lancer en local

```bash
cd api
python -m venv venv && source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Documentation interactive : http://localhost:8000/docs

> `scikit-learn` est volontairement figé (`==1.9.1`) : c'est la version qui a
> produit les `.pkl`. Une autre version peut donner des résultats faux ou planter.

## Endpoints

| Méthode | Route            | Rôle                                      |
|---------|------------------|-------------------------------------------|
| GET     | `/health`        | État de l'API et du modèle                |
| GET     | `/model/info`    | Features, modalités acceptées, seuil      |
| POST    | `/predict`       | Prédiction pour une livraison             |
| POST    | `/predict/batch` | Prédictions pour une liste (max 1000)     |

Exemple :

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "dateCommande": "2025-01-10", "datePrevue": "2025-01-20",
  "transporteur": "Service Rapide", "modeExpedition": "Economique",
  "region": "Toliara", "categorieProduit": "Electronique",
  "quantite": 3, "distanceKm": 940
}'
# {"retard":true,"probabiliteRetard":0.8677,"seuil":0.5}
```

L'API calcule elle-même `delaiPrevuJours`, `jourSemaine` et `mois` à partir des deux dates.
Une valeur hors modalités connues (ex. un nouveau transporteur) renvoie une erreur `422`.

## Configuration (variables d'environnement)

| Variable        | Défaut                                        |
|-----------------|-----------------------------------------------|
| `MODEL_DIR`     | `<racine du dépôt>/models`                    |
| `SEUIL_RETARD`  | `0.5`                                         |
| `MAX_BATCH`     | `1000`                                        |
| `CORS_ORIGINS`  | `http://localhost:3000,http://localhost:5173` |

## Tests

```bash
cd api && pytest
```

Le test `test_fidelite_avec_le_notebook` vérifie que l'API donne les mêmes
probabilités que le pipeline du notebook sur de vraies lignes du dataset.

## Docker (depuis la racine du dépôt)

```bash
docker build -t api-retards .
docker run -p 8000:8000 api-retards
```
