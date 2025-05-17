from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_recommend_endpoint_basic():
    response = client.get("/recommend?title=Interstellar&top_n=3")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 3
    assert "title" in data[0]

def test_recommend_with_genre_filter():
    response = client.get("/recommend?title=Interstellar&top_n=3&genre_filter=Science%20Fiction")
    assert response.status_code == 200
    data = response.json()
    for movie in data:
        assert "Science Fiction" in movie["genres"]
