from dotenv import load_dotenv
load_dotenv()

import os
import requests

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

if not TMDB_API_KEY:
    raise RuntimeError("TMDB_API_KEY not found. Did you forget to create .env?")


def search_movie(title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": title}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"[ERROR] TMDb search failed: {response.status_code}, {response.text}")
        return []

    data = response.json()
    return data.get("results", [])

def get_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": TMDB_API_KEY}
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"[ERROR] TMDb details failed: {response.status_code}, {response.text}")
        return {}

    return response.json()
