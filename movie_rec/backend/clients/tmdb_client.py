from dotenv import load_dotenv
load_dotenv()

import os
import requests
from functools import lru_cache

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY not found.")

@lru_cache(maxsize=100)
def search_movie(title: str):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"[ERROR] TMDb search failed: {response.status_code}, {response.text}")
        return []

    return response.json().get("results", [])

@lru_cache(maxsize=200)
def get_movie_details(movie_id: int):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"[ERROR] TMDb details failed: {response.status_code}, {response.text}")
        return {}

    return response.json()
