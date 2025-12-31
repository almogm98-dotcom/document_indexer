import os
from dotenv import load_dotenv
from google import genai

from db import get_conn, update_embedding

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"
#EMBED_MODEL = "gemini-embedding-DOES_NOT_EXIST" Its only an error test


def fetch_chunks_without_embedding(limit: int = 50) -> list[tuple[int, str]]:
    """
    Fetch chunks that don't have an embedding yet.
    Treats NULL / {} / [] as "missing".
    Raises RuntimeError with a clear message if DB query fails.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, chunk_text
            FROM document_chunks
            WHERE embedding IS NULL
               OR embedding = '{}'::jsonb
               OR embedding = '[]'::jsonb
            ORDER BY id
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        cur.close()
        return rows
    except Exception as e:
        raise RuntimeError(f"Database query failed while fetching chunks: {e}") from e
    finally:
        conn.close()


def embed_text(client: genai.Client, text: str) -> list[float]:
    """
    Create embedding using the same model as search.py.
    Raises RuntimeError with a clear message if Gemini fails or response is unexpected.
    """
    try:
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=text,
        )
    except Exception as e:
        raise RuntimeError(f"Gemini embed_content failed: {e}") from e

    try:
        vec = resp.embeddings[0].values
    except Exception as e:
        raise RuntimeError(f"Gemini returned unexpected embedding response: {e}") from e

    if not isinstance(vec, list) or len(vec) == 0:
        raise RuntimeError("Gemini returned empty/invalid embedding vector")

    try:
        return [float(x) for x in vec]
    except Exception as e:
        raise RuntimeError(f"Gemini embedding vector contains non-numeric values: {e}") from e


def main() -> int:
    try:
        key = os.getenv("GEMINI_API_KEY")
        print("GEMINI_API_KEY loaded:", "YES" if key else "NO")
        if not key:
            print("ERROR: Missing GEMINI_API_KEY in .env")
            return 1

        client = genai.Client(api_key=key)

        rows = fetch_chunks_without_embedding(limit=50)
        print(f"Found {len(rows)} chunks without embedding")

        if not rows:
            return 0

        updated = 0
        skipped = 0

        for chunk_id, chunk_text in rows:
            # guard: empty or None text
            if not chunk_text or not str(chunk_text).strip():
                print(f"Skipped chunk_id={chunk_id} (empty chunk_text)")
                skipped += 1
                continue

            vec = embed_text(client, chunk_text)

            try:
                update_embedding(chunk_id, vec)
            except Exception as e:
                raise RuntimeError(f"DB update failed for chunk_id={chunk_id}: {e}") from e

            updated += 1
            print(f"Updated chunk_id={chunk_id} dim={len(vec)}")

        print(f"Done. Updated={updated} Skipped={skipped}")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
