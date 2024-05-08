from app.tasks.celery_app import celery_app
from app.db import get_db_connection
from app.tasks.ml.TextEmbeddings import TextEmbeddings, models  # noqa: F401
from app.tasks.ml.RecSysModel import RecSysModel
from app.repository.movies import MoviesRepository

import pandas as pd
import numpy as np
import torch

import warnings
warnings.filterwarnings("ignore")

PATH = "app/tasks/ml/model.pth"

@celery_app.task(name="gen_recs")
def gen_recs():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA доступен")
    else:
        device = torch.device("cpu")  # noqa: F841
        print("CUDA недоступен, используется CPU")
        
    conn = get_db_connection()
    
    # query = '''
    #     select movies.id, movies.name, movies.year, array_agg(genres.name) as movie_genres from movies
    #     left join movies_genres on movies.id = movies_genres.movie_id
    #     left join genres on movies_genres.genre_id = genres.id
    #     group by movies.id;
    # '''

    # df_movies = pd.read_sql(query, conn)
    # df_movies['year'] = pd.to_datetime(df_movies['year']).dt.year
    
    # df_movies["movie_genres"] = df_movies["movie_genres"].apply(lambda x: [item if item else 'Отсутствуют' for item in x])
    
    # df_movies['combined'] = df_movies.apply(
    #     lambda x: f"Название: {x['name']}. Жанры: {', '.join(x['movie_genres'])}. Год: {x['year']}. Описание: {x['description']}",
    #     axis=1
    # )
    
    # df_movies = df_movies[["id", "combined"]]

    # text_embeddings = TextEmbeddings(True, True, device)
    # df_movies = text_embeddings.add_many_embeddings(df_movies, 'combined', models).drop(columns=["combined"])
    
    df_movies = pd.read_parquet("app/tasks/ml/preproccesed_movies.parquet")
    
    embedding_size = df_movies["combined_features"][0].shape[0]
    
    query = "select user_id, datetime_ as \"datetime\", duration, movie_id from logs"
    df_logs = pd.read_sql(query, conn)
    
    df_logs['datetime'] = pd.to_datetime(df_logs['datetime'])
    df_logs.sort_values(by=['user_id', 'datetime'], ascending=[True, True], inplace=True)
    df_logs = df_logs.groupby(by=["user_id", "movie_id"])["datetime"].last().reset_index()
    
    df_logs = df_logs.merge(df_movies, left_on="movie_id", right_on="id")[["datetime", "user_id", "combined_features", "movie_id"]]
    
    df_logs.sort_values(by=['user_id', 'datetime'], ascending=[True, False], inplace=True)
    dataset = df_logs.groupby("user_id").agg({
        "combined_features": lambda x: x[:20].tolist()[::-1],
        "movie_id": lambda x: x[:20].tolist()[::-1]
    }).reset_index()
    
    dataset = dataset[dataset["combined_features"].apply(lambda x: len(x) == 20)]
    
    model = RecSysModel(embedding_size, device=device)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.to(device)
    model.eval()
    
    for index, row in dataset.iterrows():
        if index < 100:
            continue
        user_id = row["user_id"]  # noqa: F841
        _, weights, ids = model.predict(row["combined_features"], df_movies)
        for w, i in zip(weights, ids):
            sub_recs = [row["movie_id"][idx] for idx in np.argpartition(w, -3)[-3:]]  # noqa: F841
            MoviesRepository.add_recs(user_id, i, sub_recs)
        break
    
    