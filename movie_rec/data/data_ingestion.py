def build_basic_movie_dataset():
    movies = pd.read_csv("data/raw/movies.csv")
    
    # Extract title and year from MovieLens
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)", expand=False)
    movies["clean_title"] = movies["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    
    # Combine title and genres for TF-IDF
    movies["metadata"] = movies["clean_title"] + " " + movies["genres"].fillna("")

    print(movies[["movieId", "clean_title", "metadata"]].head())
    movies.to_csv("data/processed/movies_basic.csv", index=False)

def main():
    movies, _ = load_movielens_data()
    enriched = enrich_movies(movies)
    enriched.to_csv("data/processed/movies_enriched.csv", index=False)
    print("✅ Enriched TMDb data saved.")
