from app.db import get_db_connection


class MoviesRepository:
    
    @classmethod
    def get_movies(cls, name: str = ""):
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = """
                        select movies.id, movies.name, movies.year, array_agg(genres.name) as movie_genres from movies
                        left join movies_genres on movies.id = movies_genres.movie_id
                        left join genres on movies_genres.genre_id = genres.id
                        where LOWER(movies.name) like %s
                        group by movies.id;
                    """
                    
                    cur.execute(query, ('%' + name.lower() + '%',))
                    movies = cur.fetchall()
                    return movies
        finally: 
            conn.close()
    
    @classmethod    
    def get_movie_by_id(cls, id: int):
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = """
                        select movies.id, movies.name, movies.year, movies.description,
                            array_agg(DISTINCT genres.name) as movie_genres,
                            array_agg(DISTINCT countries.name) as movie_countries,
                            array_agg(DISTINCT array[staff.name, staff.role]) as movie_staff
                        from movies
                        left join movies_genres on movies.id = movies_genres.movie_id
                        left join genres on movies_genres.genre_id = genres.id
                        left join Movies_Countries on movies.id = Movies_Countries.movie_id
                        left join countries on Movies_Countries.country_id = countries.id
                        left join movies_staff on movies.id = movies_staff.movie_id
                        left join staff on movies_staff.staff_id = staff.id
                        where movies.id = %s
                        group by movies.id;
                    """
                    
                    cur.execute(query, (id,))
                    movies = cur.fetchone()
                    return movies
        finally: 
            conn.close()
            
    @classmethod
    def get_user_history(cls, user_id: int):
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = '''
                        SELECT movies.id, movies.name, movies."year", logs.datetime_,
                            array_agg(DISTINCT genres.name) as movie_genres
                        FROM logs
                        left join movies on logs.movie_id = movies.id
                        left join movies_genres on movies.id = movies_genres.movie_id
                        left join genres on movies_genres.genre_id = genres.id
                        where logs.user_id = %s
                        group by movies.id, logs.id
                        order by logs.datetime_ DESC;
                    '''
                    
                    cur.execute(query, (user_id,))
                    movies = cur.fetchall()
                    return movies
        finally: 
            conn.close()
            
    @classmethod
    def add_recs(cls, user_id: int, movie_id: int, sub_rec_ids: list[int]):
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = '''DELETE FROM recs WHERE user_id = %s'''
                    cur.execute(query, (user_id,))
                    query_rec = '''INSERT INTO recs (movie_id, user_id) values (%s, %s) RETURNING id'''
                    cur.execute(query_rec, (movie_id, user_id))
                    rec_id = cur.fetchone()[0]
                    query_sub_recs = '''INSERT INTO sub_recs (rec_id, movie_id) VALUES (%s, %s)'''
                    data_to_insert = list(map(lambda x: (rec_id, x), sub_rec_ids))
                    cur.executemany(query_sub_recs, data_to_insert)
                    conn.commit()
        finally: 
            conn.close()
    
    @classmethod
    def get_recs_by_user(cls, user_id: int):
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = '''
                        SELECT recs.id as rec_id, movies.id, movies.name, movies.year,
                            array_agg(genres.name) as movie_genres
                        FROM recs
                        LEFT JOIN movies ON recs.movie_id = movies.id
                        left join movies_genres on movies.id = movies_genres.movie_id
                        left join genres on movies_genres.genre_id = genres.id
                        where recs.user_id = %s
                        group by recs.id, movies.id;
                    '''
                    
                    cur.execute(query, (user_id,))
                    movies = cur.fetchall()
                    return movies
        finally: 
            conn.close()
            
    @classmethod
    def get_rec_by_id(cls, user_id: int, rec_id: int):
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = '''
                        SELECT recs.id as rec_id, movies.id, movies.name, movies.year, movies.description,
                            array_agg(DISTINCT genres.name) as movie_genres,
                            array_agg(DISTINCT countries.name) as movie_countries,
                            array_agg(DISTINCT array[staff.name, staff.role]) as movie_staff,
                            array_agg(DISTINCT sub_recs.movie_id) as sub_ids
                        FROM recs
                        LEFT JOIN movies ON recs.movie_id = movies.id
                        left join movies_genres on movies.id = movies_genres.movie_id
                        left join genres on movies_genres.genre_id = genres.id
                        left join Movies_Countries on movies.id = Movies_Countries.movie_id
                        left join countries on Movies_Countries.country_id = countries.id
                        left join movies_staff on movies.id = movies_staff.movie_id
                        left join staff on movies_staff.staff_id = staff.id
                        left join sub_recs on recs.id = sub_recs.rec_id
                        where recs.user_id = %s AND recs.id = %s
                        group by recs.id, movies.id;
                    '''
                    
                    cur.execute(query, (user_id, rec_id))
                    movie = cur.fetchone()
                    
                    if movie is None:
                        return movie
                    
                    sub_recs = [cls.get_movie_by_id(index) for index in movie.sub_ids]
                    
                    return (movie, sub_recs)
        finally: 
            conn.close()