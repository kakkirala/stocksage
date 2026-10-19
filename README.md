# 📈 StockSage — NSE Stock RAG

Ask questions about NSE stock prices in plain English, answered from real market data.

## Why this exists

I fine-tuned a 0.5B LLM (LoRA) on NSE stock data. It learned the *format* of answers but hallucinated the *facts* — it claimed 3MINDIA "closed at 45.0" on 2017-06-28 when the real close was **13266.7**.

So I rebuilt the same data as a retrieval system. Closed-book exam → open-book exam. Now every answer is grounded in a real row from the dataset, and the UI shows you the retrieved rows as proof.

## How it works
CSV row → English sentence → embedding (all-MiniLM-L6-v2)
→ Qdrant vector store → retrieve top-3 → Ollama (llama3.2) answers
→ FastAPI web UI (StockSage)


Each CSV row becomes one faithful sentence: *"On 2017-01-23, 3MINDIA opened at 12500.0, high 12500.0, low 12230.15, closed at 12299.45, volume 357."* — nothing more.

## Data

- NSE daily stock data, **January 2017**: 35,896 records
- Source: [minatverma/nse-stocks-data](https://www.kaggle.com/datasets/minatverma/nse-stocks-data) on Kaggle (download `FINAL_FROM_DF.csv` into this folder)

## Honest limitations

- **January 2017 only** — the full 2-year slice was too heavy for my laptop's CPU; the pipeline scales, the hardware didn't.
- **It retrieves, it doesn't predict.** Asking "what will SBIN do tomorrow?" gets you history, not a forecast. Past prices don't predict future ones — I verified that separately with an honest backtest.
- **Small LLM, small fumbles.** llama3.2 occasionally misreads a retrieved row; a stricter system prompt fixed the cases I hit.

## Run it

```powershell
# 1. Start Qdrant
docker compose up -d   # qdrant on localhost:6333

# 2. Install
python -m pip install qdrant-client sentence-transformers ollama fastapi uvicorn pandas

# 3. Ingest (a few minutes on CPU)
python ingest_full.py

# 4. Ask in the terminal
python answer.py

# 5. Or open the web UI
uvicorn app:app --reload
# → http://127.0.0.1:8000

