from __future__ import annotations

from pathlib import Path
import sys

from docx import Document
from pypdf import PdfReader

from dotenv import load_dotenv
load_dotenv()


def extract_text_from_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts).strip()


def extract_text_from_docx(path: Path) -> str:
    doc = Document(str(path))
    parts: list[str] = []
    for para in doc.paragraphs:
        t = (para.text or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts).strip()


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(path)
    if suffix == ".docx":
        return extract_text_from_docx(path)

    raise ValueError(f"Unsupported file type: {suffix}. Only .pdf / .docx are supported.")

def chunk_fixed_size(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

        if start < 0:
            start = 0

    return chunks

import re


def chunk_by_sentences(text: str, max_chars: int = 300) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', cleaned)

    chunks: list[str] = []
    current = ""

    for s in sentences:
        s = s.strip()
        if not s:
            continue

        if len(s) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(s), max_chars):
                part = s[i:i + max_chars].strip()
                if part:
                    chunks.append(part)
            continue

        if current and (len(current) + 1 + len(s)) > max_chars:
            chunks.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip() if current else s

    if current:
        chunks.append(current.strip())

    return chunks

import psycopg2
import os

def insert_chunk(chunk_text: str, filename: str, split_strategy: str):
    conn = psycopg2.connect(
        host=os.getenv("PGHOST"),
        port=os.getenv("PGPORT"),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
    )

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO document_chunks (chunk_text, embedding, filename, split_strategy)
        VALUES (%s, %s, %s, %s)
        """,
        (chunk_text, "{}", filename, split_strategy)
    )

    conn.commit()
    cur.close()
    conn.close()


def main() -> int:

    import os
    key = os.getenv("GEMINI_API_KEY")
    print("GEMINI_API_KEY loaded:", "YES" if key else "NO")

    if len(sys.argv) < 2:
        print("Usage: python index_documents.py <path_to_pdf_or_docx>")
        return 1

    file_path = Path(sys.argv[1]).expanduser().resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return 1

    text = extract_text(file_path)
    chunks = chunk_fixed_size(text)

    print(f"\nCreated {len(chunks)} fixed-size chunks:\n")
    for i, c in enumerate(chunks, start=1):
        print(f"--- Chunk {i} ---")
        print(c)
        print()

    sentence_chunks = chunk_by_sentences(text)

    print(f"\nCreated {len(sentence_chunks)} sentence-based chunks:\n")
    for i, c in enumerate(sentence_chunks, start=1):
        print(f"=== Sentence Chunk {i} ===")
        print(c)
        print()

    for c in chunks:
        insert_chunk(
            chunk_text=c,
            filename=file_path.name,
            split_strategy="fixed_size"
        )

    print("\n--- Extracted text (first 1000 chars) ---\n")
    print(text[:1000])
    print("\n--- End ---\n")

    if not text.strip():
        print("Warning: No text extracted (file may be scanned image PDF).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
