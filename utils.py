import surprise as sp
import logging
from functools import wraps
import time
import numpy as np
from functools import lru_cache
logger = logging.getLogger(__name__)

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Function {func.__name__} took {end_time - start_time} seconds to execute")
        return result
    return wrapper

class Recommender:
    """
    Recommender class to recommend articles to a user
    Allow an hybrid recommendation based on content and collaborative filtering
    Use embeddings to compute the similarity between the user and the articles
    """
    def __init__(self, articles, articles_clicks_per_user, embeddings, model, min_rating=5):
        self.articles = articles
        self.articles_clicks_per_user = articles_clicks_per_user
        self.idx_map = {i: articles.loc[i, "article_id"] for i in range(len(articles))}
        self.embeddings = embeddings
        self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.model = model
        # Filter to only keep the articles with a rating greater than 3
        self.articles_clicks_per_user = self.articles_clicks_per_user[self.articles_clicks_per_user.rating>=min_rating]
        logger.info(f"Number of articles: {len(self.articles_clicks_per_user)}")

    def get_top_articles(self, id_user,k:int=5, hybrid=True, alpha=0.7,similarity=True):
        logger.info(f"Getting top {k} articles for user {id_user}")
        if hybrid:  
            result = self.score_hybride(id_user, alpha)
        elif similarity:
            result = self.score_collaboratif(id_user)
        else:
            result = self.score_contenu(id_user)
        top = sorted(result.items(), key=lambda x: x[1], reverse=True)[:k]
        top = sorted(top, key=lambda x: x[0])
        article_details = self.articles.loc[self.articles["article_id"].isin([i for i,_ in top])].sort_values(by='article_id')
        article_details['score'] = [i[1] for i in top]
        return article_details.sort_values(by='score', ascending=False)
    
    def train(self):
        svd = sp.SVD(n_factors=100,biased=True)
        self.model = svd
        self.model.fit(self.articles_clicks_per_user)
    
    @property
    def all_items(self):
        return self.articles["article_id"].unique()

    @lru_cache(maxsize=1)
    @timer
    def seen(self, id_user):
        return set(self.articles_clicks_per_user.loc[self.articles_clicks_per_user.user_id==id_user, "click_article_id"])

    @timer
    def score_contenu(self, id_user):
        candidates = [id_ for id_ in self.articles["article_id"] if id_ not in self.seen(id_user)]
        candidates_embeddings = self.embeddings[candidates]
        user_embedding = self.embeddings[list(self.seen(id_user))]
        scores = candidates_embeddings@user_embedding.T
        return {i: score for i, score in zip(candidates, scores.max(axis=1))}

    @timer
    def score_collaboratif(self, id_user):
        return {i: self.model.predict(id_user,i).est for i in self.all_items if i not in self.seen(id_user)}

    def score_hybride(self, id_user, alpha=0.7):
        svd_scores = self.score_collaboratif(id_user)
        contenu_scores = self.score_contenu(id_user)
        return {i: alpha*svd_scores[i] + (1-alpha)*contenu_scores[i] for i in svd_scores}
        
        

