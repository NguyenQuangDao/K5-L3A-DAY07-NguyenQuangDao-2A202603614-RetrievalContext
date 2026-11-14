"""Retrieval adapters for Day 7: Dense (pgvector Day 6), Lexical (TF-IDF), and Hybrid RRF."""
import os
import sys
from pathlib import Path
from typing import Any, cast

import psycopg
from openai import OpenAI
from psycopg.rows import dict_row

# Allow importing starter/baseline.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "starter"))
from baseline import TfidfIndex

DEFAULT_DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
DEFAULT_STRATEGY = "strategy_b"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class DenseRetriever:
    """Query the Day 6 pgvector store using OpenAI text-embedding-3-small."""

    def __init__(
        self,
        database_url: str | None = None,
        strategy: str = DEFAULT_STRATEGY,
        embed_model: str = DEFAULT_EMBED_MODEL,
        api_key: str | None = None,
    ) -> None:
        self.database_url: str = database_url or os.getenv("DATABASE_URL") or DEFAULT_DB_URL
        self.strategy = strategy
        self.embed_model = embed_model
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY chưa được thiết lập.")
        self.client = OpenAI(api_key=key)
        self.total_tokens = 0

    def embed_query(self, query: str) -> list[float]:
        resp = self.client.embeddings.create(
            model=self.embed_model,
            input=[query.strip() or " "],
        )
        if getattr(resp, "usage", None):
            self.total_tokens += resp.usage.total_tokens or 0
        return resp.data[0].embedding

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        vec = self.embed_query(query)
        sql = """
            SELECT
                chunk_id AS id,
                document_id AS source_id,
                source,
                page,
                text,
                metadata,
                (1 - (embedding <=> %s::vector)) AS score
            FROM document_chunks
            WHERE strategy = %s
        """
        params: list[Any] = [str(vec), self.strategy]
        if filters:
            for k, v in filters.items():
                sql += " AND metadata->>%s = %s"
                params.extend([str(k), str(v)])
        sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s;"
        params.extend([str(vec), top_k])

        with psycopg.connect(self.database_url, row_factory=dict_row) as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cast(list[dict[str, Any]], cur.fetchall())

        return [{**row, "score": float(row["score"]), "strategy": "dense"} for row in rows]


class LexicalRetriever:
    """TF-IDF lexical retriever over exported Day 6 chunks."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        candidates = self.index.search(query, top_k=len(self.index.chunks))
        if filters:
            candidates = [
                c
                for c in candidates
                if all(str(c.get("metadata", {}).get(k)) == str(v) for k, v in filters.items())
            ]
        return [{**c, "strategy": "tfidf"} for c in candidates[:top_k]]


class HybridRRFRetriever:
    """Combine Dense (pgvector) and Lexical (TF-IDF) using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: DenseRetriever,
        lexical: LexicalRetriever,
        *,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        self.dense = dense
        self.lexical = lexical
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        dense_res = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        lex_res = self.lexical.search(query, filters=filters, top_k=self.candidate_k)

        scores: dict[str, float] = {}
        by_id: dict[str, dict[str, Any]] = {}

        for rank, item in enumerate(dense_res, start=1):
            cid = item["id"]
            by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        for rank, item in enumerate(lex_res, start=1):
            cid = item["id"]
            if cid not in by_id:
                by_id[cid] = item
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        return [{**by_id[cid], "score": score, "strategy": "hybrid_rrf"} for cid, score in ranked]


def pack_context_by_score(
    results: list[dict[str, Any]], budget: int = 2400
) -> tuple[str, list[str]]:
    """Deduplicate by id, sort by score descending, and pack within character budget."""
    unique: dict[str, dict[str, Any]] = {}
    for c in results:
        cid = c["id"]
        if cid not in unique or c.get("score", 0.0) > unique[cid].get("score", 0.0):
            unique[cid] = c

    ordered = sorted(unique.values(), key=lambda c: (-c.get("score", 0.0), c["id"]))
    blocks: list[str] = []
    ids: list[str] = []
    for c in ordered:
        block = f"[{c['id']} | {c['source']} | page {c['page']}]\n{c['text']}"
        candidate_text = "\n\n".join(blocks + [block])
        if len(candidate_text) <= budget:
            blocks.append(block)
            ids.append(c["id"])
    return "\n\n".join(blocks), ids

