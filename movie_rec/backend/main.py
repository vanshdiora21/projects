from fastapi import FastAPI, Query
from backend.api.routes import router

app = FastAPI(title="Movie Recommendation API")
app.include_router(router)


@app.get("/recommend")
def recommend(
    title: str,
    top_n: int = 5,
    sort_by: str = Query("popularity", enum=["popularity", "vote_average"]),
    genre_filter: str = Query(None, description="Optional genre filter (e.g., 'Science Fiction')")
):
    results = search_movie(title)
    if not results:
        return {"input": title, "recommendations": []}

    enriched = []
    for movie in results:
        details = get_movie_details(movie["id"])
        genres = [g["name"] for g in details.get("genres", [])]

        # Apply genre filter if specified
        if genre_filter and genre_filter not in genres:
            continue

        enriched.append({
            "title": movie["title"],
            "overview": movie.get("overview"),
            "genres": genres,
            "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
            "popularity": movie.get("popularity", 0),
            "vote_average": movie.get("vote_average", 0)
        })

    # Sort based on selected criteria
    enriched.sort(key=lambda x: x.get(sort_by, 0), reverse=True)

    return {
        "input": title,
        "recommendations": enriched[:top_n]
    }
