from pydantic import BaseModel
from typing import List, Optional

class Movie(BaseModel):
    title: str
    overview: Optional[str]
    genres: List[str]
    poster: Optional[str]
    popularity: float
    vote_average: float
