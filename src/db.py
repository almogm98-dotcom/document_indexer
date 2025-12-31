import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

load_dotenv()


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )


def test_db_connection():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
    tables = cur.fetchall()
    cur.close()
    conn.close()
    return tables


def update_embedding(chunk_id: int, embedding: list[float]):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE document_chunks
        SET embedding = %s
        WHERE id = %s
        """,
        (Json(embedding), chunk_id)
    )
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    tables = test_db_connection()
    print("Tables in database:")
    for t in tables:
        print(t[0])

    update_embedding(1, [0.1, 0.2, 0.3])
    print("Embedding updated")
