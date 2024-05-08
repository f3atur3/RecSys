from app.config import settings
import psycopg2
from psycopg2.extras import NamedTupleCursor


def get_db_connection():
    conn = psycopg2.connect(
        dbname=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASS,
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        cursor_factory=NamedTupleCursor
    )
    return conn
