import pytest
from recommender.content_based import ContentBasedRecommender

@pytest.fixture
def mock_data():
    return {
        'titles': ['Inception', 'The Matrix', 'Interstellar'],
        'overviews': [
            'A thief who steals corporate secrets through the use of dream-sharing technology.',
            'A computer hacker learns about the true nature of reality and his role in the war.',
            'A team of explorers travel through a wormhole in space.'
        ]
    }

def test_recommender_load(mock_data):
    rec = ContentBasedRecommender(mock_data['titles'], mock_data['overviews'])
    assert rec is not None

def test_recommend_output(mock_data):
    rec = ContentBasedRecommender(mock_data['titles'], mock_data['overviews'])
    results = rec.recommend("Inception", top_n=2)
    assert isinstance(results, list)
    assert len(results) == 2
