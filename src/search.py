from __future__ import annotations

import math
import os
import sys
from typing import Any

import psycopg2
from dotenv import load_dotenv
from google import genai

load_dotenv()

EMBED_MODEL = "gemini-embedding-001"


def get_conn():
    """
    Create a DB connection.
    Raises RuntimeError with a clear message on failure.
    """
    try:
        return psycopg2.connect(
            host=os.getenv("PGHOST"),
            port=os.getenv("PGPORT"),
            dbname=os.getenv("PGDATABASE"),
            user=os.getenv("PGUSER"),
            password=os.getenv("PGPASSWORD"),
        )
    except Exception as e:
        raise RuntimeError(f"Database connection failed: {e}") from e


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("-inf")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / denom if denom != 0.0 else float("-inf")


def embed_text(client: genai.Client, text: str) -> list[float]:
    """
    Embed query using the SAME embedding model used by embedder.py.
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

    # Force float conversion (also validates numeric contents)
    try:
        return [float(x) for x in vec]
    except Exception as e:
        raise RuntimeError(f"Gemini embedding vector contains non-numeric values: {e}") from e


def fetch_chunks_with_embeddings() -> list[dict[str, Any]]:
    """
    Fetch only chunks with non-empty embeddings.
    Raises RuntimeError with a clear message if DB query fails.
    """
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, filename, split_strategy, chunk_text, embedding
            FROM document_chunks
            WHERE embedding IS NOT NULL
              AND embedding <> '{}'::jsonb
              AND embedding <> '[]'::jsonb
            ORDER BY id
            """
        )
        rows = cur.fetchall()
        cur.close()
    except Exception as e:
        raise RuntimeError(f"Database query failed while fetching chunks: {e}") from e
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r[0],
                "filename": r[1],
                "split_strategy": r[2],
                "chunk_text": r[3],
                "embedding": r[4],  # jsonb -> usually list in Python
            }
        )
    return out


def main() -> int:
    try:
        if len(sys.argv) < 2:
            print('Usage: python src/search.py "your query" [top_k]')
            return 1

        query = sys.argv[1].strip()
        if not query:
            print("ERROR: query is empty")
            return 1

        # Parse top_k safely
        top_k = 5
        if len(sys.argv) >= 3:
            try:
                top_k = int(sys.argv[2])
            except ValueError:
                print("ERROR: top_k must be an integer")
                return 1
            if top_k <= 0:
                print("ERROR: top_k must be > 0")
                return 1

        key = os.getenv("GEMINI_API_KEY")
        print("GEMINI_API_KEY loaded:", "YES" if key else "NO")
        if not key:
            print("ERROR: Missing GEMINI_API_KEY in .env")
            return 1

        client = genai.Client(api_key=key)

        query_emb = embed_text(client, query)
        print("query dim =", len(query_emb))

        chunks = fetch_chunks_with_embeddings()
        if not chunks:
            print("No chunks with embeddings found in database.")
            return 0

        scored: list[tuple[float, dict[str, Any]]] = []
        skipped = 0

        for c in chunks:
            emb = c.get("embedding")

            if not isinstance(emb, list) or len(emb) == 0:
                skipped += 1
                continue

            try:
                emb_vec = [float(x) for x in emb]
            except Exception:
                skipped += 1
                continue

            score = cosine_similarity(query_emb, emb_vec)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)

        print(f"\nTop {top_k} results:\n")
        for score, c in scored[:top_k]:
            preview = (c.get("chunk_text") or "").replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:220] + "..."
            print(
                f"id={c['id']} score={score:.4f} file={c['filename']} strategy={c['split_strategy']}"
            )
            print(preview)
            print()

        if skipped:
            print(f"(Skipped {skipped} rows with invalid embeddings)")

        return 0

    except Exception as e:
        # Graceful failure for submission: clear message + non-zero exit
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
