"""Build cases.json (20 answerable from Day 6 + 4 unanswerable for Day 7)."""
import json
from pathlib import Path

QUESTIONS_PATH = Path("starter/data/converted/questions.json")
CASES_PATH = Path("starter/data/converted/cases.json")

UNANSWERABLE = [
    {
        "id": "u_dev_1",
        "split": "dev",
        "question": "What is the exact genome size in base pairs of Amoeba proteus?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Sách Animal Life (1900) không đề cập đến kích thước bộ gen hay base pairs.",
    },
    {
        "id": "u_dev_2",
        "split": "dev",
        "question": "How is CRISPR-Cas9 used to edit genes in Protozoa according to the book?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Công nghệ CRISPR-Cas9 không tồn tại trong tài liệu.",
    },
    {
        "id": "u_test_1",
        "split": "test",
        "question": "What is the IUCN Red List conservation status of the Bengal tiger?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Tài liệu không chứa thông tin phân loại bảo tồn IUCN Red List.",
    },
    {
        "id": "u_test_2",
        "split": "test",
        "question": "Which electron microscope magnification is required to view mitochondria in Amoeba?",
        "answerable": False,
        "expected_behavior": "search_and_abstain",
        "absence_reason": "Tài liệu chỉ nhắc kính hiển vi quang học (compound microscope), không có kính hiển vi điện tử hay ty thể.",
    },
]


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    cases = [
        {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
        }
        for q in questions
    ] + UNANSWERABLE

    CASES_PATH.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Đã tạo {len(cases)} cases -> {CASES_PATH}")


if __name__ == "__main__":
    main()

