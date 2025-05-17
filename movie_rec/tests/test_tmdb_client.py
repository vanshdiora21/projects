import pytest
from backend.clients.tmdb_client import search_movie, get_movie_details

def test_search_movie_returns_results():
    results = search_movie("Interstellar")
    assert isinstance(results, list)
    assert len(results) > 0
    assert "title" in results[0]

def test_get_movie_details_structure():
    results = search_movie("Inception")
    movie_id = results[0]["id"]
    details = get_movie_details(movie_id)

    assert isinstance(details, dict)
    assert "genres" in details
    assert "title" in details
