import os
import pandas as pd

BASE_PATH = "earnings_calls"


def parse_quarter(quarter_str: str):
    year, q = quarter_str.split("-year/")[0], quarter_str.split("/")[-1][0]
    month = {'1': 1, '2': 4, '3': 7, '4': 10}[q]
    return pd.Timestamp(f"{year}-{month:02d}-01")


def get_earnings_call_text(url: str):
    """Scrape transcript text from Roic.ai earnings call page."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1200,800")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        driver.get(url)
        body_text = driver.find_element("tag name", "body").text

        # Try to isolate transcript section more cleanly
        if "Earnings Call Transcript" in body_text:
            transcript = body_text.split("Earnings Call Transcript", 1)[-1]
        else:
            transcript = body_text

        # Cut off footer if present
        if "Footer" in transcript:
            transcript = transcript.split("Footer", 1)[0]

        transcript = transcript.strip()

        if len(transcript) < 500:  # heuristic: too short = probably not valid transcript
            print(f"⚠️ Transcript too short or invalid at {url}")
            return None

        return transcript
    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return None
    finally:
        driver.quit()


def get_year_quarters_from_dates(start_date: pd.Timestamp, end_date: pd.Timestamp):
    year_quarters = []
    current = pd.Timestamp(start_date.year, ((start_date.month - 1) // 3) * 3 + 1, 1)
    end = pd.Timestamp(end_date.year, ((end_date.month - 1) // 3) * 3 + 1, 1)

    while current <= end:
        q = (current.month - 1) // 3 + 1
        year_quarters.append(f"{current.year}-year/{q}-quarter")
        # move to next quarter
        month = current.month + 3
        year = current.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        current = pd.Timestamp(year, month, 1)
    return year_quarters

def scrape_ticker(ticker: str, start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Incrementally scrape transcripts for a ticker.
    - Loads existing CSV if available
    - Finds only missing quarters in the requested range
    - Scrapes & appends them
    """
    year_quarters = get_year_quarters_from_dates(start_date, end_date)
    ticker_dir = f"./earnings_calls/{ticker}/"
    os.makedirs(ticker_dir, exist_ok=True)
    file_path = f"{ticker_dir}scraped_earnings_calls.csv"

    # Load existing data if present
    if os.path.exists(file_path):
        existing = pd.read_csv(file_path)
    else:
        existing = pd.DataFrame(columns=["year_quarter", "earnings_call_raw_text", "ticker", "date"])

    already_have = set(existing["year_quarter"].unique())
    missing_quarters = [yq for yq in year_quarters if yq not in already_have]

    if not missing_quarters:
        print(f"✅ All requested quarters for {ticker} already scraped.")
        return existing

    print(f"🔎 Scraping {len(missing_quarters)} new transcripts for {ticker}...")

    base_url = f"https://www.roic.ai/quote/{ticker}/transcripts/"
    new_calls = []
    for yq in missing_quarters:
        url = f"{base_url}{yq}"
        txt = get_earnings_call_text(url)
        if txt:  # only add if scrape succeeded
            new_calls.append({
                "year_quarter": yq,
                "earnings_call_raw_text": txt,
                "ticker": ticker,
                "date": parse_quarter(yq),
            })
        else:
            print(f"⚠️ Skipped {ticker} {yq} (no transcript)")

    if not new_calls:
        print(f"⚠️ No new transcripts successfully scraped for {ticker}.")
        return existing

    new_df = pd.DataFrame(new_calls)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Deduplicate just in case
    combined = combined.drop_duplicates(subset=["ticker", "year_quarter"]).sort_values("date")

    # Only write if something changed
    if len(combined) > len(existing):
        combined.to_csv(file_path, index=False)
        print(f"💾 Saved updated transcripts for {ticker}")

    return combined


def combine_all_calls(start_date=None, end_date=None):
    """
    Combine all per-ticker scraped earnings calls into one DataFrame.
    Optionally filter by date range.
    """
    files = glob.glob("earnings_calls/*/scraped_earnings_calls.csv")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if "date" not in df:
            continue
        df['date'] = pd.to_datetime(df['date'])
        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    all_calls = pd.concat(dfs, ignore_index=True)
    all_calls = all_calls.drop_duplicates(subset=["ticker", "year_quarter"]).sort_values(['ticker', 'date']).reset_index(drop=True)

    if start_date:
        all_calls = all_calls[all_calls['date'] >= pd.Timestamp(start_date)]
    if end_date:
        all_calls = all_calls[all_calls['date'] <= pd.Timestamp(end_date)]

    all_calls.to_csv("all_calls.csv", index=False)
    return all_calls
