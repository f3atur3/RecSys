from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination

from app.api.routers import all_routers
from app.tasks.scheduler import gen_recs  # noqa: F401

app = FastAPI()

for router in all_routers:
    app.include_router(router)

# @app.get("/get_recs")
# def get_recs():
#     gen_recs.delay()

add_pagination(app)

origins = [
    "http://frontend:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
