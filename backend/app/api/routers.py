from app.api.users import router as users_router
from app.api.movies import router as movies_router


all_routers = [
    users_router,
    movies_router
]