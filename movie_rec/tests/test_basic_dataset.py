
import pandas as pd
from recommender.content_based import ContentBasedRecommender

def test_offline_metadata_recommender():
    df = pd.read_csv("data/processed/movies_basic.csv")
    rec = ContentBasedRecommender(df)
    result = rec.recommend("Toy Story", top_n=3)
    assert isinstance(result, list)
