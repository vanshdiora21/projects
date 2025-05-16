import os
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def search_movie(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(url, params=params)
    return response.json()["results"]

def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)
    return response.json()
