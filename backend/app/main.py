from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

class MyModel(BaseModel):
    id: int
    name: str

@app.get("/data", response_model=List[MyModel])
async def get_data():
    # Получение данных от psycopg2
    data = [{"id": 1, "name": "John"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Alice"}]
    return data

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
