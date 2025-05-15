from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedRecommender:
    def __init__(self, titles, overviews):
        self.titles = titles
        self.overviews = overviews

        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(overviews)

        # Store index → title map for lookups
        self.index_to_title = {i: title for i, title in enumerate(titles)}
        self.title_to_index = {title: i for i, title in enumerate(titles)}

    def recommend(self, title, top_n=5):
        if title not in self.title_to_index:
            return []

        idx = self.title_to_index[title]
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()

        # Get top N excluding the movie itself
        similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
        return [self.index_to_title[i] for i in similar_indices]
