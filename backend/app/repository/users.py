from app.db import get_db_connection


class UsersRepository:
    
    @classmethod
    def get_user_by_username(cls, username: str):
        conn = get_db_connection()
        try:
            with conn:
                with conn.cursor() as cur:
                    query = """SELECT id, name, email, login, password
                        FROM users
                        WHERE login = %s;
                    """
                    
                    cur.execute(query, (username,))
                    user = cur.fetchone()
                    
                    return user
        finally: 
            conn.close()