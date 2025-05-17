from fastapi import APIRouter, Query
from typing import List, Optional, Literal
from backend.clients.tmdb_client import search_movie, get_movie_details
from backend.models.movie import Movie

router = APIRouter()


@router.get("/recommend", response_model=List[Movie])
def recommend(
    title: str,
    top_n: int = Query(5, ge=1, le=20, description="Number of recommendations to return"),
    sort_by: Literal["popularity", "vote_average"] = Query("popularity", description="Sort recommendations by this field"),
    genre_filter: Optional[str] = Query(None, description="Optional genre filter (e.g. 'Action', 'Science Fiction')")
):
    results = search_movie(title)
    if not results:
        return []

    enriched = []

    for movie in results:
        try:
            details = get_movie_details(movie["id"])
        except Exception:
            continue  # Skip on TMDb fetch failure

        genres = [g["name"] for g in details.get("genres", [])]
        if genre_filter and genre_filter not in genres:
            continue

        enriched.append(Movie(
            title=movie.get("title", "Unknown"),
            overview=movie.get("overview", ""),
            genres=genres,
            poster=f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
            popularity=movie.get("popularity", 0.0),
            vote_average=movie.get("vote_average", 0.0)
        ))

    enriched.sort(key=lambda x: getattr(x, sort_by, 0), reverse=True)
    return enriched[:top_n]
