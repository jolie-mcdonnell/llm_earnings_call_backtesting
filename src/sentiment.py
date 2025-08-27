import os
import json
import pandas as pd
import time
from openai import OpenAI
from typing import List


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

ROOT_PROGRESS_PATH = "all_calls_progress.csv"

MODEL = "gpt-5-nano"
MAX_OUTPUT_TOKENS = 500
CHAR_CAP = 80_000
SAVE_EVERY = 20  # save frequently, but smaller than before for safety

PROMPT_HEADER = """I will provide the transcript of an earnings call. Your job is to analyze the text only based on what is actually present in the transcript. For each of the following categories, assign a score between -1 and 1:

forward_looking_sentiment: How positive or negative is the company’s outlook or projections for the future?
management_confidence: How confident does management appear about business performance and strategy?
risk_and_uncertainty: How much concern, risk, or uncertainty is conveyed (higher = more risk)?
qa_sentiment: How positive or negative is the tone during the Q&A section with analysts?
opening_sentiment: How positive or negative is the opening section or prepared remarks?
financial_performance_sentiment: Based solely on what is said in the transcript, how positively is past financial performance portrayed?
macroeconomic_reference_sentiment: If there are references to broader macroeconomic conditions, how optimistic or pessimistic are those?

If a category is not addressed clearly in the transcript, return exactly 0 for that category.

Use the following format for your output:
{
  "forward_looking_sentiment": ___,
  "management_confidence": ___,
  "risk_and_uncertainty": ___,
  "qa_sentiment": ___,
  "opening_sentiment": ___,
  "financial_performance_sentiment": ___,
  "macroeconomic_reference_sentiment": ___
}
Do not include any text or explanation—only return the JSON object. Do not guess or infer information that is not directly stated in the transcript.

Transcript:"""


# def build_prompt(transcript: str) -> str:
#     return f"{PROMPT_HEADER}\n{(transcript or '')[:CHAR_CAP]}"

# def call_gpt_nano(prompt: str, max_retries: int = 5):
#     delays = [1, 2, 5, 10, 20]
#     for attempt in range(max_retries):
#         try:
#             resp = client.responses.create(
#                 model=MODEL,
#                 input=prompt,
#                 max_output_tokens=MAX_OUTPUT_TOKENS,
#                 reasoning={"effort": "low"},
#                 text={"format": {"type": "json_object"}, "verbosity": "low"},
#             )
#             return resp.output_text.strip()
#         except Exception:
#             if attempt == max_retries - 1:
#                 return None
#             time.sleep(delays[min(attempt, len(delays)-1)])

# def safe_json_load(s: str):
#     if not s: return {}
#     try: return json.loads(s)
#     except:
#         s2 = s.strip().strip("`").replace("```json","").replace("```","").strip()
#         try: return json.loads(s2)
#         except: return {}

# def analyze_sentiment(ticker: str):
#     ticker_dir = f"earnings_calls/{ticker}"
#     scraped_path = os.path.join(ticker_dir, "scraped_earnings_calls.csv")
#     processed_path = os.path.join(ticker_dir, "processed_earnings_calls.csv")

#     # --- Load scraped data (with transcripts) ---
#     if not os.path.exists(scraped_path):
#         raise FileNotFoundError(f"No scraped data found for {ticker}")
#     scraped_df = pd.read_csv(scraped_path)

#      # --- Load or init processed data (NO transcripts) ---
#     if os.path.exists(processed_path):
#         processed_df = pd.read_csv(processed_path)
#     else:
#         processed_df = pd.DataFrame(columns=[
#             "date","ticker","url","analysis_json",
#             "forward_looking_sentiment","management_confidence",
#             "risk_and_uncertainty","qa_sentiment","opening_sentiment",
#             "financial_performance_sentiment","macroeconomic_reference_sentiment"
#         ])

#     # --- Figure out which calls are NEW (exist in scraped but not processed) ---
#     already_processed_dates = set(processed_df["date"].astype(str))
#     new_calls = scraped_df[~scraped_df["date"].astype(str).isin(already_processed_dates)]

#     if new_calls.empty:
#         print(f"✅ No new calls to process for {ticker}")
#         return processed_df

#     result_cols = [
#         "forward_looking_sentiment","management_confidence",
#         "risk_and_uncertainty","qa_sentiment","opening_sentiment",
#         "financial_performance_sentiment","macroeconomic_reference_sentiment"
#     ]

#     processed_since_save = 0
#     for _, row in new_calls.iterrows():
#         prompt = build_prompt(row["earnings_call_raw_text"])
#         txt = call_gpt_nano(prompt)
#         parsed = safe_json_load(txt)

#         new_row = {
#             "date": row["date"],
#             "ticker": ticker,
#             "url": row.get("url",""),
#             "analysis_json": txt,
#         }
#         for c in result_cols:
#             new_row[c] = parsed.get(c, None)

#         processed_df = pd.concat([processed_df, pd.DataFrame([new_row])], ignore_index=True)

#         processed_since_save += 1
#         if processed_since_save >= SAVE_EVERY:
#             processed_df.to_csv(processed_path, index=False)
#             processed_since_save = 0

#     # Final save
#     processed_df.to_csv(processed_path, index=False)
#     print(f"✅ Processed {len(new_calls)} new calls for {ticker}")
#     return processed_df


# ---------------------- Helpers ----------------------
def build_prompt(transcript: str) -> str:
    return f"{PROMPT_HEADER}\n{(transcript or '')[:CHAR_CAP]}"

def call_gpt_nano(prompt: str, max_retries: int = 5):
    delays = [1, 2, 5, 10, 20]
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                reasoning={"effort": "low"},
                text={"format": {"type": "json_object"}, "verbosity": "low"},
            )
            return resp.output_text.strip()
        except Exception as e:
            print(f"⚠️ OpenAI call failed (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(delays[min(attempt, len(delays)-1)])

def safe_json_load(s: str):
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        s2 = str(s).strip().strip("`").replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(s2)
        except Exception:
            return {}

def _result_cols() -> List[str]:
    return [
        "forward_looking_sentiment",
        "management_confidence",
        "risk_and_uncertainty",
        "qa_sentiment",
        "opening_sentiment",
        "financial_performance_sentiment",
        "macroeconomic_reference_sentiment",
    ]

# ---------------------- Main (DataFrame-based) ----------------------
def analyze_sentiment(all_calls: pd.DataFrame) -> pd.DataFrame:
    """
    Incrementally analyze sentiment for a combined DataFrame of transcripts.

    Expected columns in `all_calls`:
      - ticker (str)
      - date (datetime-like or str)
      - year_quarter (str like '2023-year/1-quarter')
      - earnings_call_raw_text (str transcript)

    Side effects:
      - Writes per-ticker processed files at: earnings_calls/{ticker}/processed_earnings_calls.csv
        (NO transcripts are stored here)
      - Writes/updates a global consolidated ROOT_PROGRESS_PATH (no transcripts)
    Returns:
      - A consolidated DataFrame of processed rows (no transcripts)
    """
    required = {"ticker", "date", "year_quarter", "earnings_call_raw_text"}
    missing = [c for c in required if c not in all_calls.columns]
    if missing:
        raise ValueError(f"analyze_sentiment: missing required columns: {missing}")

    # Normalize types
    df = all_calls.copy()
    df["ticker"] = df["ticker"].astype(str)
    df["year_quarter"] = df["year_quarter"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Container to build the consolidated file at the end
    consolidated_rows = []
    total_new = 0
    processed_since_save = 0

    # Process per ticker to use per-ticker processed cache
    for ticker, tdf in df.groupby("ticker"):
        ticker_dir = os.path.join("earnings_calls", ticker)
        os.makedirs(ticker_dir, exist_ok=True)
        processed_path = os.path.join(ticker_dir, "processed_earnings_calls.csv")

        # Load existing processed (no transcripts)
        if os.path.exists(processed_path):
            proc = pd.read_csv(processed_path, dtype={"year_quarter": str})
            if not proc.empty:
                proc["date"] = pd.to_datetime(proc["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        else:
            proc = pd.DataFrame(columns=["date", "ticker", "year_quarter", "url", "analysis_json"] + _result_cols())

        # Keys that identify a call
        processed_keys = set(zip(proc.get("year_quarter", pd.Series(dtype=str)).astype(str),
                                 proc.get("date", pd.Series(dtype=str)).astype(str)))

        # Prepare current ticker rows
        tdf = tdf.copy()
        tdf["url"] = tdf.get("url", "")

        new_entries = []

        for _, row in tdf.iterrows():
            key = (row["year_quarter"], row["date"])
            if key in processed_keys:
                # already done; skip the API call
                continue

            text = row.get("earnings_call_raw_text")
            if not isinstance(text, str) or not text.strip():
                print(f"⚠️ Skipping {ticker} {row['year_quarter']} ({row['date']}) — empty transcript.")
                continue

            prompt = build_prompt(text)
            txt = call_gpt_nano(prompt)
            parsed = safe_json_load(txt)

            entry = {
                "date": row["date"],
                "ticker": ticker,
                "year_quarter": row["year_quarter"],
                "url": row.get("url", ""),
                "analysis_json": txt,
            }
            for c in _result_cols():
                entry[c] = parsed.get(c, None)

            new_entries.append(entry)
            total_new += 1
            processed_since_save += 1

            # Periodic save for safety
            if processed_since_save >= SAVE_EVERY:
                if new_entries:
                    proc = pd.concat([proc, pd.DataFrame(new_entries)], ignore_index=True)
                    proc = proc.drop_duplicates(subset=["ticker", "year_quarter", "date"]).sort_values(["ticker", "date"])
                    proc.to_csv(processed_path, index=False)
                    new_entries.clear()

                # Also refresh the global consolidated file
                _write_global_progress(ROOT_PROGRESS_PATH)
                processed_since_save = 0

        # Final save for this ticker (append any remaining new entries)
        if new_entries:
            proc = pd.concat([proc, pd.DataFrame(new_entries)], ignore_index=True)

        # Dedup & save per ticker
        if not proc.empty:
            proc = proc.drop_duplicates(subset=["ticker", "year_quarter", "date"]).sort_values(["ticker", "date"])
            proc.to_csv(processed_path, index=False)

        # After finishing this ticker, extend consolidated list with all rows from proc
        consolidated_rows.extend(proc.to_dict(orient="records"))

    # Build and write consolidated global file
    consolidated_df = pd.DataFrame(consolidated_rows)
    if not consolidated_df.empty:
        consolidated_df = consolidated_df.drop_duplicates(subset=["ticker", "year_quarter", "date"]).sort_values(["ticker", "date"])
        consolidated_df.to_csv(ROOT_PROGRESS_PATH, index=False)

    print(f"✅ Sentiment analysis complete. New calls processed: {total_new}")
    # Return consolidated results (no transcripts)
    return consolidated_df if not consolidated_df.empty else pd.DataFrame(
        columns=["date", "ticker", "year_quarter", "url", "analysis_json"] + _result_cols()
    )

def _write_global_progress(global_path: str):
    """
    Rebuild the global consolidated CSV from per-ticker processed files.
    """
    rows = []
    base = "earnings_calls"
    if not os.path.isdir(base):
        return
    for ticker in os.listdir(base):
        p = os.path.join(base, ticker, "processed_earnings_calls.csv")
        if os.path.exists(p):
            dfp = pd.read_csv(p, dtype={"year_quarter": str})
            if not dfp.empty:
                dfp["date"] = pd.to_datetime(dfp["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                rows.append(dfp)
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["ticker", "year_quarter", "date"]).sort_values(["ticker", "date"])
    out.to_csv(global_path, index=False)