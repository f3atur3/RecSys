from datetime import date, datetime
from typing import Annotated, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi_pagination import Page, paginate
from pydantic import BaseModel

from app.repository.movies import MoviesRepository
from app.api.users import User
from app.utils.auth import get_current_user


class MovieInList(BaseModel):
    id: int
    name: str
    year: date
    movie_genres: list[Optional[str]]
    
class Movie(MovieInList):
    description: str
    movie_countries: list[Optional[str]]
    movie_staff: list[list[Optional[str]]]
    
class MovieInHistory(MovieInList):
    datetime_: datetime
    
class MovieInRecs(Movie):
    rec_id: int
    
class MovieRec(BaseModel):
    movie: MovieInRecs
    sub_recs: list[Movie]

router = APIRouter(
    prefix="/movies",
    tags=["Movies"]
)

@router.get("/")
def get_movies(name: str = "") -> Page[MovieInList]:
    movies = MoviesRepository.get_movies(name)
    return paginate(movies)

@router.get("/id/{id}")
def get_movie_by_id(id: int) -> Optional[Movie]:
    movie = MoviesRepository.get_movie_by_id(id)
    
    if movie is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such movie"
        )
    
    return movie

@router.get("/history")
def get_user_history(
    current_user: Annotated[User, Depends(get_current_user)]
) -> Page[MovieInHistory]:
    movies = MoviesRepository.get_user_history(current_user.id)
    return paginate(movies)

@router.get("/recs")
def get_user_recs(
    current_user: Annotated[User, Depends(get_current_user)]
) -> List[MovieInRecs]:
    movies = MoviesRepository.get_recs_by_user(current_user.id)
    return movies

@router.get("/rec/id/{rec_id}")
def get_rec_by_id(
    current_user: Annotated[User, Depends(get_current_user)],
    rec_id: int
) -> MovieRec:
    dataset = MoviesRepository.get_rec_by_id(current_user.id, rec_id)
    if dataset is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No such recomend for current user"
        )
    
    movie, sub_recs = dataset
    
    return {"movie": movie, "sub_recs": sub_recs}
