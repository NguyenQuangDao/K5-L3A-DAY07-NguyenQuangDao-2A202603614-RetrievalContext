# Báo cáo Day 7 — Nguyễn Quang Đạo (2A202603614)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tài liệu từ Day 6: *Animal Life: A First Book of Zoology* (David Starr Jordan & Vernon L. Kellogg, 1900, phạm vi Public Domain; file PDF gốc nằm ở repo Day 6).
  - Tập chunk sử dụng: `strategy_b` (tách theo đoạn văn `by_paragraph`, `chunk_size=800`) gồm **40 chunks** (các trang chương I, trang 22–48), đã export sang [chunks.json](../../data/converted/chunks.json).
- **Vector store, embedding model, phiên bản và số chiều:**
  - Vector store: PostgreSQL 16 + `pgvector:0.8.7` (container Day 6 `day6_student_starter-postgres-1`, port `55433`, bảng `document_chunks`, lọc `strategy = 'strategy_b'`).
  - Embedding model: OpenAI `text-embedding-3-small`, số chiều **1536**, độ đo khoảng cách cosine (`<=>`).
- **Agent model, cách chạy và phân chia dev/test:**
  - Agent LLM: OpenAI `gpt-4o-mini` (`temperature=0`, `response_format={"type": "json_object"}`), chạy qua [agent_runner.py](../../src/agent_runner.py) và [agentic.py](../../starter/agentic.py).
  - Bộ benchmark gồm **24 câu** tại [cases.json](../../data/converted/cases.json) và [questions.json](../../data/converted/questions.json):
    - **Dev (17 câu):** 15 câu có đáp án (`q01`–`q15`) + 2 câu thiếu bằng chứng (`u_dev_1`, `u_dev_2`).
    - **Test (7 câu):** 5 câu có đáp án (`q16`–`q20`) + 2 câu thiếu bằng chứng (`u_test_1`, `u_test_2`).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án được kiểm tra đối chiếu chuỗi `gold_evidence` khớp nguyên văn trong chunk đúng `source_id` và `page` qua hàm `validate()` của [baseline.py](../../starter/baseline.py).
  - Câu thiếu bằng chứng hỏi các khái niệm sinh học hiện đại không có trong sách năm 1900 (kích thước bộ gen theo base pair, CRISPR-Cas9, danh lục IUCN Red List, kính hiển vi điện tử/ty thể), có ghi `absence_reason` trong `cases.json`.
  - Tiêu chí chốt cấu hình trên `dev`: ưu tiên `Context Evidence Rate` và `Hit@1`/`Evidence Hit@3` cao nhất với chi phí và độ trễ chấp nhận được; không dùng tập `test` để tinh chỉnh.

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | `tfidf` (Lexical), `top_k=3`, no filter, FIFO `budget=800` chars | Baseline offline không cần vector DB | dev | 9/15 (60.0%) | 11/15 (73.3%) | 1/15 (6.7%) | 0.05 ms / \$0 |
| Phương pháp 2 | `dense` (pgvector `text-embedding-3-small`), `top_k=3`, no filter, FIFO `budget=800` chars | Dense semantic bắt đồng nghĩa tốt hơn TF-IDF | dev | 13/15 (86.7%) | 13/15 (86.7%) | 1/15 (6.7%) | 328.1 ms / \$3.28e-6 |
| Phương pháp 3 | `hybrid_rrf` (Dense + TF-IDF, RRF $k=60$, `candidate_k=10`), `top_k=3`, FIFO `budget=800` chars | Hợp nhất hạng từ khóa và ngữ nghĩa để tăng độ phủ | dev | 11/15 (73.3%) | 13/15 (86.7%) | 1/15 (6.7%) | 289.1 ms / \$3.28e-6 |
| Cải thiện 1 | `dense`, `top_k=3`, `sort_by_score=True`, `budget=2400` chars | Tăng `budget` từ 800 lên 2400 để không cắt mất chunk `by_paragraph` | dev | 13/15 (86.7%) | 13/15 (86.7%) | 12/15 (80.0%) | 506.4 ms / \$3.28e-6 |
| Cải thiện 2 | `hybrid_rrf`, `top_k=3`, `sort_by_score=True`, `budget=2400` chars | Thử Hybrid RRF với `budget=2400` | dev | 11/15 (73.3%) | 13/15 (86.7%) | 10/15 (66.7%) | 376.9 ms / \$3.28e-6 |
| Cải thiện 3 (Chốt) | `dense`, `top_k=3`, `sort_by_score=True`, `budget=4000` chars | Tăng `budget` lên 4000 để giữ được cả các chunk dài >2400 ký tự (như `q04`) | dev | 13/15 (86.7%) | 13/15 (86.7%) | 13/15 (86.7%) | 244.2 ms / \$3.28e-6 |
| Baseline | `tfidf`, `top_k=3`, FIFO `budget=800` chars | Đánh giá độc lập trên `test` | test | 2/5 (40.0%) | 4/5 (80.0%) | 1/5 (20.0%) | 0.06 ms / \$0 |
| Cấu hình đã chọn | `dense`, `top_k=3`, `sort_by_score=True`, `budget=4000` chars | Đánh giá độc lập trên `test` sau khi chốt trên `dev` | test | 3/5 (60.0%) | 4/5 (80.0%) | 3/5 (60.0%) | 4282.5 ms / \$1.40e-6 |

### Phân tích 2 lỗi trên `dev` và lý do chọn cấu hình

1. **Lỗi 1 — Context bị cắt bỏ do `budget=800` và `budget=2400` (`q04`):**
   - *Hiện tượng:* Trên `dev`, `dense` đạt `Hit@1 = 13/15`, nhưng với `budget=800` thì `Context Evidence Rate` chỉ đạt `1/15`. Khi tăng lên `budget=2400`, chỉ số tăng lên `12/15`, vẫn sót `q04` dù `q04` đã đạt `Hit@1` (`animal_life_zoology_strategy_b_p25_idx0_7cd05e9c`).
   - *Nguyên nhân:* Các chunk `by_paragraph` dài hơn 800 ký tự, riêng chunk trang 25 dài hơn 2400 ký tự nên `pack_context` bỏ qua chunk top-1 và chèn chunk ngắn hơn ở trang 24 vào context.
   - *Cách xử lý & kết quả:* Tăng `budget=4000` ký tự và sắp xếp theo score giảm dần (`sort_by_score=True`), giúp `Context Evidence Rate` trên `dev` tăng từ `1/15` -> `12/15` -> **`13/15` (86.7%)**.
2. **Lỗi 2 — Retrieval miss dẫn đến từ chối nhầm (`q02`) và ranh giới trang (`q12`):**
   - *Hiện tượng:* `q02` (*"What is required to see the microscopic animals in stagnant water?"*, evidence `"compound microscope"` ở trang 23) không vào top-3 ở cả `dense` lẫn `hybrid_rrf`. Khi chạy Agent trên `dev`, Agent đã search 2 lượt với query viết lại nhưng vẫn không kéo được trang 23 nên trả về `abstain`. Với `q12`, `gold` nằm ở đầu trang 33 trong khi retriever trả về trang 32 (đoạn trước đó bị ngắt trang).
   - *Nguyên nhân:* Chunk trang 23 có nhiều nhiễu OCR từ sách scan năm 1900 nên vector similarity thấp hơn các trang mô tả sinh vật chung chung; đồng thời `by_paragraph` chưa gộp đoạn vắt ngang ranh giới trang.
- **Lý do chốt cấu hình:** Chọn `dense` (`text-embedding-3-small`) + `budget=4000` + `sort_by_score=True` vì đạt `Hit@1` (`13/15`) và `Context Evidence Rate` (`13/15`) cao nhất trên `dev` (vượt `hybrid_rrf` vốn bị TF-IDF kéo giảm `Hit@1` xuống `11/15`), trong khi chi phí embedding chỉ `\$3.28e-6` cho toàn bộ 15 câu `dev`.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Controller: `OpenAIController` (`gpt-4o-mini`, `temperature=0`, JSON output) gọi tool tìm kiếm thông qua `run_agent` trong [agentic.py](../../starter/agentic.py).
  - Điều kiện dừng: Agent trả về `action="answer"` kèm citation `[chunk-id]` hợp lệ (hoặc `citations=[]` để `abstain` khi thiếu bằng chứng), hoặc chạm ngưỡng `max_searches` (`1` cho Baseline, `3` cho Cấu hình chọn).
  - Xử lý lỗi: Lỗi API/DB được bắt trong [agent_runner.py](../../src/agent_runner.py) và ghi riêng `status="error"` kèm thông báo lỗi, không thay bằng kết quả giả.

### Bảng tổng hợp Agent (trên `test`, `n = 7`: 5 có đáp án + 2 thiếu bằng chứng)

| Cấu hình / split | Tool use | Số lượt gọi (TB) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline (`dense`, `budget=800`, `max_searches=1`) / `test` | 7/7 (100%) | 1.00 | 0: 4, 1: 0, 2: 1 (TB: **0.40/2**) | 1/2 (50.0%) | 2/2 (100%) | 2/2 (100%) | 3/5 (60.0%) |
| Cấu hình đã chọn (`dense`, `budget=4000`, `max_searches=3`) / `test` | 7/7 (100%) | 1.29 | 0: 2, 1: 0, 2: 3 (TB: **1.20/2**) | 3/4 (75.0%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20.0%) |

*(Chấm context quality cho 5 câu `test` có đáp án: Baseline đạt `0: 4, 1: 0, 2: 1`; Cấu hình đã chọn đạt `0: 2, 1: 0, 2: 3`.)*

### Chi tiết audit từng câu `test` (Cấu hình đã chọn)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt search; context chứa chunk `p38_idx0` (đúng evidence) | Trả lời đúng cách pseudopodia đan xen tạo mạng lưới protoplasmic, trích `[...p38_idx0_07ba5faf]` | `answer_grade=2`, `citation_support=[true]`, `context=2`. Evidence hỗ trợ đầy đủ. |
| `q17` | Có đáp án | 1 lượt search; context chỉ có `p38_idx0` (thiếu trang 39 chứa đáp án về `nucleus`) | Trả lời *"lies within the central capsule"*, trích `[...p38_idx0_07ba5faf]` | `answer_grade=0`, `citation_support=[false]`, `context=0`. **Lỗi hallucination**: `p38_idx0` chỉ nói capsule chia protoplasm thành 2 phần, không hề nhắc đến `nucleus`; model tự suy đoán và gắn ID có trong context. |
| `q18` | Có đáp án | 1 lượt search; context chứa `p40_idx0` (đúng evidence) | Trả lời đúng hóa thạch Foraminifera có thể truy vết xa trong lịch sử địa chất, trích `[...p40_idx0_1421c032]` | `answer_grade=2`, `citation_support=[true]`, `context=2`. Evidence hỗ trợ đầy đủ. |
| `q19` | Có đáp án | 1 lượt search; không truy xuất được trang 45 | `"Không đủ bằng chứng để trả lời."`, `citations=[]` | `answer_grade=0`, `citation_support=[]`, `context=0`. Từ chối nhầm (false abstention) do retriever miss. |
| `q20` | Có đáp án | 1 lượt search; context chứa `p48_idx0` (đúng evidence) | Trả lời đúng 2 tế bào hợp nhất (fuse) tạo màng cellulose và bước vào resting stage, trích `[...p48_idx0_5316adee]` | `answer_grade=2`, `citation_support=[true]`, `context=2`. Evidence hỗ trợ đầy đủ. |
| `u_test_1` | Thiếu evidence | 3 lượt search mở rộng query về IUCN Red List & Bengal tiger, không có kết quả phù hợp | `"Không đủ bằng chứng để trả lời."`, `citations=[]` | `abstention_correct=true`. Agent tìm kiếm kỹ 3 lượt rồi từ chối chính xác, không bịa đặt. |
| `u_test_2` | Thiếu evidence | 1 lượt search về kính hiển vi điện tử và ty thể | `"Không đủ bằng chứng để trả lời."`, `citations=[]` | `abstention_correct=true`. Từ chối chính xác. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Traces & summary retrieval: `results/retrieval/{dev-tfidf,dev-dense-800,dev-hybrid-800,dev-dense-2400,dev-hybrid-2400,dev-dense-4000,test-tfidf,test-dense-4000}/`
  - Traces, reviews & summary agent:
    - Dev: `results/agent/dev-dense-2400/traces.json` (17 câu, tổng chi phí `\$0.003840`, median `4111.5 ms`)
    - Test Baseline: `results/agent/test-baseline/{traces.json,reviews.json,summary.json}` (7 câu, tổng chi phí `\$0.000882`, median `2669.6 ms`)
    - Test Chosen: `results/agent/test-chosen/{traces.json,reviews.json,summary.json}` (7 câu, tổng chi phí `\$0.002245`, median `4089.7 ms`)
  - Lệnh tái lập:
    ```bash
    /opt/anaconda3/bin/python3 src/benchmark_retrieval.py --chunks data/converted/chunks.json --questions data/converted/questions.json --split test --method dense --budget 4000 --sort-by-score --output results/retrieval/test-dense-4000
    /opt/anaconda3/bin/python3 src/agent_runner.py --chunks data/converted/chunks.json --cases data/converted/cases.json --split test --method dense --budget 4000 --max-searches 3 --top-k 3 --output results/agent/test-chosen/traces.json
    /opt/anaconda3/bin/python3 starter/agent_metrics.py --cases data/converted/cases.json --traces results/agent/test-chosen/traces.json --reviews results/agent/test-chosen/reviews.json --split test --output results/agent/test-chosen/summary.json
    ```
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:**
  - **Trước/sau:** Tăng `budget` từ `800` lên `4000` và cho phép `max_searches=3` giúp giảm tỷ lệ từ chối nhầm trên `test` từ **3/5 (60%) xuống 1/5 (20%)**, tăng điểm Grounded answer trung bình từ **0.40/2 lên 1.20/2**, và tăng Citation hợp lệ từ **1/2 (50%) lên 3/4 (75%)**, trong khi vẫn giữ **Abstention đúng 2/2 (100%)**.
  - **Đánh đổi:** Chi phí trên tập `test` tăng từ `\$0.000882` lên `\$0.002245` (+2.5x) và độ trễ trung vị tăng từ `2.67s` lên `4.09s` do context dài hơn và `u_test_1` thực hiện 3 lượt search.
  - **Giới hạn:** (1) Chunking `by_paragraph` từ Day 6 có một số đoạn dài (>2400 ký tự) hoặc ngắt ngang trang (`q12`, `q17`) và nhiễu OCR từ sách scan năm 1900; (2) Kiểm tra citation tự động chỉ kiểm tra sự tồn tại của ID chứ chưa ngăn được LLM suy đoán kiến thức ngoài chunk như ở `q17`, do đó vẫn cần audit thủ công.
