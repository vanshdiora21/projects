import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, dataframe):
        self.df = dataframe
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.df["metadata"] = (
        self.df["overview"].fillna("") + " " +
        self.df["genres"].apply(lambda g: " ".join(eval(g)) if pd.notna(g) else "") + " " +
        self.df["cast"].apply(lambda c: " ".join(eval(c)) if pd.notna(c) else "") + " " +
        self.df["director"].fillna("")
        )

        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["metadata"])

        self.index_to_title = self.df["clean_title"].to_dict()
        self.title_to_index = {title: i for i, title in self.index_to_title.items()}

    def recommend(self, title, top_n=5):
        if title not in self.title_to_index:
            return []

        idx = self.title_to_index[title]
        cosine_sim = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        similar_indices = cosine_sim.argsort()[::-1][1:top_n+1]
        return [self.index_to_title[i] for i in similar_indices]
