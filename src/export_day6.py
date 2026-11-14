import json
from pathlib import Path
from typing import Any, cast

import psycopg
from psycopg.rows import dict_row

DB_URL = "postgresql://day6:day6_local_only@127.0.0.1:55433/day6_lab"
STRATEGY = "strategy_b"

DAY6_QUESTIONS = Path(
    "../../LAB-DAY6/K5-L3A-DAY06-2A202603614-DataIngestion/starter/benchmark_questions.json"
)
OUT_DIR = Path("starter/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with psycopg.connect(DB_URL, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, document_id, source, page, text, metadata
            FROM document_chunks
            WHERE strategy = %s
            ORDER BY page, chunk_index
            """,
            (STRATEGY,),
        )
        chunks = cast(list[dict[str, Any]], cur.fetchall())

    if not chunks:
        raise RuntimeError(f"Không tìm thấy chunk nào cho strategy='{STRATEGY}'")

    default_doc_id = chunks[0]["document_id"]
    (OUT_DIR / "day6_chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Đã export {len(chunks)} chunks -> {OUT_DIR / 'day6_chunks.json'}")

    raw_qs = json.loads(DAY6_QUESTIONS.read_text(encoding="utf-8"))
    out_qs = []
    for q in raw_qs:
        item = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "gold_source_id": q.get("gold_source_id", default_doc_id),
            "gold_page": q["gold_page"],
            "evidence_quote": q["gold_evidence"],
            "expected_answer": q.get("expected_answer") or q["gold_evidence"],
        }
        if "filters" in q:
            item["filter"] = q["filters"]
        out_qs.append(item)

    (OUT_DIR / "day6_questions.json").write_text(
        json.dumps(out_qs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Đã export {len(out_qs)} câu hỏi -> {OUT_DIR / 'day6_questions.json'}")


if __name__ == "__main__":
    main()

