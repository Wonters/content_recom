import azure.functions as func
import logging
import json
from azure.storage.blob import BlobServiceClient
import pandas as pd
import numpy as np
from utils import Recommender
from functools import lru_cache
import pickle
import io
import os 

# # Configuration du logging
logging.getLogger('azure.storage.blob').setLevel(logging.WARNING)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


# Configuration de la connexion au Blob Storage
connect_str = os.environ["AzureWebJobsStorage"]
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
container_name = "uploads"  # Nom du conteneur où seront stockés les fichiers

# Chargement des données nécessaires
container_client = blob_service_client.get_container_client(container_name)

def download_blob(blob_name):
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()

# Charger les données depuis le blob storage
@lru_cache(maxsize=1)
def load_data():
    articles_content = download_blob("articles.parquet")
    if not articles_content:
        raise ValueError("articles.parquet is empty or does not exist")
    articles = pd.read_parquet(io.BytesIO(articles_content))
    logging.info("Articles loaded successfully")

    clicks_content = download_blob("clicks.parquet")
    if not clicks_content:
        raise ValueError("clicks.parquet is empty or does not exist")
    clicks_per_user = pd.read_parquet(io.BytesIO(clicks_content))
    clicks_per_user = clicks_per_user[["user_id", "click_article_id"]].assign(rating=1).groupby(["user_id", "click_article_id"], as_index=False).agg({"rating": "sum"})
    logging.info("Clicks data loaded successfully")

    embeddings_content = download_blob("embeddings.pkl")
    if not embeddings_content:
        raise ValueError("embeddings.pkl is empty or does not exist")
    embeddings = np.load(io.BytesIO(embeddings_content), allow_pickle=True)
    logging.info("Embeddings loaded successfully")

    model_content = download_blob("model.pkl")
    if not model_content:
        raise ValueError("model.pkl is empty or does not exist")
    model = pickle.loads(model_content)
    logging.info("Model loaded successfully")
        
    return articles, clicks_per_user, embeddings, model

@app.function_name(name="TestFunction") 
@app.route(route="test")
def test(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps({
            "message": "Hello, World!"
        }),
        mimetype="application/json",
        status_code=200
    )

@app.function_name(name="RecommendFunction")
@app.route(route="recommand")
def recommand(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    
    try:
        # Récupérer l'ID utilisateur
        logging.info(f"Request POST: {req.get_json()}")
        params = req.get_json()
        user_id = params.get('user_id')
        k = params.get('k', 5)
        similarity = bool(int(params.get('similarity')))
        hybrid = params.get('hybrid', True)
        alpha = params.get('alpha', 0.7)
        if not user_id:
            return func.HttpResponse(
                json.dumps({
                    "message": "User ID is required",
                    "status": "error"
                }),
                mimetype="application/json",
                status_code=400
            )
        
        # Convertir user_id en entier
        user_id = int(user_id)
        
        # Charger les données
        articles, articles_clicks_per_user,embeddings, model = load_data()
        recommender = Recommender(articles, articles_clicks_per_user, embeddings, model, min_rating=2)
        
        # Obtenir les recommandations
        recommendations = recommender.get_top_articles(user_id, k=5, hybrid=True,similarity=similarity, alpha=0.7)
        
        # Convertir les recommandations en format JSON
        recommendations_json = recommendations.to_dict(orient='records')
        
        return func.HttpResponse(
            json.dumps({
                "message": f"Recommandations pour l'utilisateur {user_id}",
                "status": "success",
                "recommendations": recommendations_json
            }),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logging.error(f"Erreur lors de la recommandation: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "message": f"Erreur lors de la recommandation: {str(e)}",
                "status": "error"
            }),
            mimetype="application/json",
            status_code=500
        )

