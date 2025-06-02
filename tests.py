from azure.storage.blob import BlobServiceClient
import pandas as pd
import io
import numpy as np
from utils import Recommender
import pickle
articles = pd.read_parquet("data/articles.parquet")
clicks = pd.read_parquet("data/clicks.parquet")
embeddings = np.load(open("data/articles_embeddings.pickle", "rb"), allow_pickle=True)
model = pickle.load(open("data/model.pkl", "rb"))
clicks = clicks[['user_id', 'click_article_id']].assign(rating=1).groupby(['user_id', 'click_article_id'], as_index=False).agg({'rating': 'sum'})
recommender = Recommender(articles, clicks, embeddings, model, min_rating=5)

def test_recommender_hybride():
    results = recommender.get_top_articles(16280, k=5, hybrid=True, similarity=True, alpha=0.7)
    print(results.to_dict(orient='records'))
    assert results.to_dict(orient='records') == [{'category_id': 136, 'created_at_ts': 1503313613000, 'publisher_id': 0, 'words_count': 172}, 
                                                 {'category_id': 299, 'created_at_ts': 1506974928000, 'publisher_id': 0, 'words_count': 176}, 
                                                 {'category_id': 323, 'created_at_ts': 1506958823000, 'publisher_id': 0, 'words_count': 221}, 
                                                 {'category_id': 289, 'created_at_ts': 1500891937000, 'publisher_id': 0, 'words_count': 149}, 
                                                 {'category_id': 7, 'created_at_ts': 1490920889000, 'publisher_id': 0, 'words_count': 203}]


def test_recommender_similarity():
    results = recommender.get_top_articles(16280, k=5, hybrid=False, similarity=True, alpha=0.7)
    print(results.to_dict(orient='records'))
    assert results.to_dict(orient='records') == [{'category_id': 7, 'created_at_ts': 1490920889000, 'publisher_id': 0, 'words_count': 203}, 
                                                 {'category_id': 7, 'created_at_ts': 1504311089000, 'publisher_id': 0, 'words_count': 179}, 
                                                 {'category_id': 7, 'created_at_ts': 1518800874000, 'publisher_id': 0, 'words_count': 215}, 
                                                 {'category_id': 7, 'created_at_ts': 1519859342000, 'publisher_id': 0, 'words_count': 231}, 
                                                 {'category_id': 7, 'created_at_ts': 1512637614000, 'publisher_id': 0, 'words_count': 292}]
    

def test_recommender_collaboratif():
    results = recommender.get_top_articles(16280, k=5, hybrid=False, similarity=False, alpha=0.7)
    print(results.to_dict(orient='records'))
    assert results.to_dict(orient='records') == [{'category_id': 289, 'created_at_ts': 1500891937000, 'publisher_id': 0, 'words_count': 149}, 
                                                 {'category_id': 299, 'created_at_ts': 1506974928000, 'publisher_id': 0, 'words_count': 176}, 
                                                 {'category_id': 136, 'created_at_ts': 1503313613000, 'publisher_id': 0, 'words_count': 172}, 
                                                 {'category_id': 327, 'created_at_ts': 1508089569000, 'publisher_id': 0, 'words_count': 197}, 
                                                 {'category_id': 323, 'created_at_ts': 1506958823000, 'publisher_id': 0, 'words_count': 221}]


def test_score_collaboratif():
    results = recommender.score_collaboratif(16280)
    assert len(results) == 5