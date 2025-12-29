import pandas as pd
import numpy as np
import os
import requests
import json
import time

# Load .env file into environment if present (simple loader)
ENV_PATH = os.path.join(os.path.dirname(__file__), '.env') if '__file__' in globals() else '.env'
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass
from datetime import datetime, timedelta

# ==============================
# CONFIG
# ==============================
PHASE1_FILE = "phase-1results/Phase1_final_robust_momentum.xlsx"
STOCK_30M_DIR = "downloaded_data/intraday_30m"

OUTPUT_DIR = "phase-2results"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "phase2_step1_2_3_4_with_catalyst.xlsx"
)

# ==============================
# API KEYS (loaded from environment / .env)
# ==============================
# NOTE: API keys must be provided via environment variables. A .env file
# has been added to the workspace with the original values for convenience.
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
MISTRAL_API_URL = os.environ.get("MISTRAL_API_URL", "https://api.mistral.ai/v1/chat/completions")
MISTRAL_MODEL = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable not set. Add it to .env or export it.")

USE_PERCENTILE_SCORING = True
MAX_VOLATILITY_SCORE = 40

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Phase-2 Step-1, Step-2 & Step-3 Started")

# ==============================
# LOAD PHASE-1 DATA
# ==============================
df_raw = pd.read_excel(PHASE1_FILE)

required_cols = ["Symbol", "ATR% Raw", "VolMult", "Above VWAP"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing required columns in Phase-1 file: {missing}")

# ==============================
# FILTER: Only process stocks with ALL "YES" criteria
# ==============================
print(f"Total stocks in Phase-1: {len(df_raw)}")

# Filter for stocks where all YES/NO columns are "YES"
yes_columns = [col for col in df_raw.columns if col.startswith("Above") or "Pass" in col or "Gate" in col]
if yes_columns:
    df_filtered = df_raw.copy()
    for col in yes_columns:
        if col in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[col] == "YES"]
    print(f"Stocks with all YES criteria: {len(df_filtered)}")
else:
    # If no YES/NO columns, take top 100 by ATR% (most volatile)
    df_filtered = df_raw.nlargest(100, "ATR% Raw")
    print(f"No YES/NO columns found. Taking top 100 by ATR%: {len(df_filtered)}")

if len(df_filtered) == 0:
    raise ValueError("No stocks passed the filter! Check your Phase-1 results.")

# ==============================
# KEEP ONLY NECESSARY COLUMNS FROM PHASE-1
# ==============================
# Essential columns to keep from Phase-1
columns_to_keep = ["Date", "Symbol", "ATR% Raw", "VolMult", "Above VWAP"]

# Check if there are any additional useful columns (optional)
optional_cols = ["Sector", "Industry", "Market Cap", "Price"]
for col in optional_cols:
    if col in df_filtered.columns:
        columns_to_keep.append(col)

# Select only necessary columns
df = df_filtered[columns_to_keep].copy()

print(f"Kept {len(columns_to_keep)} columns from Phase-1: {columns_to_keep}")
print(f"Processing {len(df)} stocks for Phase-2...\n")

# ==============================
# FIND & LOAD NIFTY 30M DATA
# ==============================
NIFTY_FILE = os.path.join(STOCK_30M_DIR, "^NSEI.csv")

if not os.path.exists(NIFTY_FILE):
    raise FileNotFoundError(
        "Missing NIFTY file: downloaded_data/intraday_30m/^NSEI.csv"
    )

nifty_df = pd.read_csv(NIFTY_FILE)

nifty_df["Datetime"] = pd.to_datetime(nifty_df["Datetime"])
nifty_df["Close"] = pd.to_numeric(nifty_df["Close"], errors="coerce")

nifty_df = nifty_df.dropna(subset=["Close"]).sort_values("Datetime")

if len(nifty_df) < 2:
    raise ValueError("Not enough valid NIFTY 30m candles")

NIFTY_now = float(nifty_df["Close"].iloc[-1])
NIFTY_30m = float(nifty_df["Close"].iloc[-2])

# ==============================
# LOAD STOCK 30M DATA
# ==============================
P_now_list = []
P_30m_list = []

for symbol in df["Symbol"]:
    stock_file = os.path.join(STOCK_30M_DIR, f"{symbol}.csv")

    if not os.path.exists(stock_file):
        P_now_list.append(np.nan)
        P_30m_list.append(np.nan)
        continue

    sdf = pd.read_csv(stock_file)
    sdf["Datetime"] = pd.to_datetime(sdf["Datetime"])
    sdf["Close"] = pd.to_numeric(sdf["Close"], errors="coerce")

    sdf = sdf.dropna(subset=["Close"]).sort_values("Datetime")

    if len(sdf) < 2:
        P_now_list.append(np.nan)
        P_30m_list.append(np.nan)
        continue

    # Date synchronization check vs NIFTY (warn if mismatch > 5 minutes)
    try:
        nifty_latest_time = pd.to_datetime(nifty_df["Datetime"].iloc[-1])
        stock_latest_time = pd.to_datetime(sdf["Datetime"].iloc[-1])
        delta_secs = abs((nifty_latest_time - stock_latest_time).total_seconds())
        if delta_secs > 300:
            print(f"WARNING: Time mismatch for {symbol}: nifty {nifty_latest_time} vs stock {stock_latest_time} (delta {int(delta_secs)}s)")
    except Exception:
        # If any parsing error, continue silently
        pass

    P_now_list.append(float(sdf["Close"].iloc[-1]))
    P_30m_list.append(float(sdf["Close"].iloc[-2]))

df["P_now"] = P_now_list
df["P_30m"] = P_30m_list
df["NIFTY_now"] = NIFTY_now
df["NIFTY_30m"] = NIFTY_30m

df = df.dropna(subset=["P_now", "P_30m"])

# ==============================
# STEP-1: VOLATILITY SCORE (0–40)
# ==============================
def volatility_score(atr, all_atr):
    pct = np.sum(all_atr <= atr) / len(all_atr)
    return round(MAX_VOLATILITY_SCORE * pct, 2)

all_atr = df["ATR% Raw"].values
df["VolatilityScore_0_40"] = [volatility_score(a, all_atr) for a in all_atr]

# ==============================
# STEP-2: VOLUME SHOCK SCORE (0–25)
# ==============================
def volume_shock(vol_mult, above_vwap):
    if vol_mult < 1.0:
        s = 0
    elif vol_mult <= 1.2:
        s = 5
    elif vol_mult <= 1.5:
        s = 10
    elif vol_mult <= 2.0:
        s = 17
    elif vol_mult <= 2.5:
        s = 22
    else:
        s = 25

    if above_vwap == "NO":
        s = min(s, 15)

    return s

df["VolumeShockScore_0_25"] = df.apply(
    lambda r: volume_shock(r["VolMult"], r["Above VWAP"]),
    axis=1
)

# ==============================
# STEP-3: RELATIVE STRENGTH vs NIFTY (0–25)
# ==============================
def rs_score(rs):
    if rs <= 0:
        return 0
    elif rs <= 0.2:
        return 5
    elif rs <= 0.5:
        return 10
    elif rs <= 1.0:
        return 17
    elif rs <= 1.5:
        return 22
    else:
        return 25

df["R_stock_30m"] = ((df["P_now"] - df["P_30m"]) / df["P_30m"]) * 100
df["R_nifty_30m"] = ((NIFTY_now - NIFTY_30m) / NIFTY_30m) * 100
df["RS_30m"] = df["R_stock_30m"] - df["R_nifty_30m"]

df["RSScore_0_25"] = df["RS_30m"].apply(rs_score)

# ==============================
# STEP-4: CATALYST SCORE (0–10)
# ==============================
def fetch_google_news_rss(symbol):
    """
    Fetch company news from Google News RSS (Free, unlimited)
    Returns list of news articles with headline, datetime, summary
    """
    try:
        import xml.etree.ElementTree as ET
        from datetime import datetime
        import re
        
        # Clean symbol for search
        clean_symbol = symbol.replace(".NS", "").replace("&", "and")
        
        # Google News RSS URL
        search_query = f"{clean_symbol}+stock+India"
        url = f"https://news.google.com/rss/search?q={search_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse RSS XML
        root = ET.fromstring(response.content)
        
        formatted_news = []
        current_time = datetime.now()
        
        # Find all items (news articles)
        for item in root.findall('.//item')[:10]:  # Limit to 10 most recent
            title_elem = item.find('title')
            pub_date_elem = item.find('pubDate')
            description_elem = item.find('description')
            
            if title_elem is not None and pub_date_elem is not None:
                headline = title_elem.text or ""
                pub_date_str = pub_date_elem.text or ""
                summary = description_elem.text if description_elem is not None else ""
                
                # Parse pubDate (RFC 2822 format)
                try:
                    # Example: "Wed, 25 Dec 2024 10:30:00 GMT"
                    from email.utils import parsedate_to_datetime
                    article_time = parsedate_to_datetime(pub_date_str)
                    
                    # Calculate minutes ago
                    time_diff = current_time - article_time.replace(tzinfo=None)
                    minutes_ago = int(time_diff.total_seconds() / 60)
                    
                    # Convert to timestamp for consistency
                    article_timestamp = int(article_time.timestamp())
                    
                except Exception as e:
                    # If parsing fails, assume it's recent (30 minutes ago)
                    minutes_ago = 30
                    article_timestamp = int((current_time.timestamp() - 1800))
                
                formatted_news.append({
                    "headline": headline,
                    "datetime": article_timestamp,
                    "summary": summary[:200],  # Limit summary length
                    "minutes_ago": minutes_ago
                })
        
        return formatted_news
    
    except Exception as e:
        print(f"Error fetching Google News for {symbol}: {str(e)}")
        return []


def analyze_news_with_llm(symbol, news_articles):
    """
    Use Mistral AI to analyze news and classify events
    Returns: eventType, impact, direction, recencyMinutes
    """
    if not news_articles:
        return {
            "eventType": "none",
            "impact": "none",
            "direction": "neutral",
            "recencyMinutes": 9999
        }
    
    # Prepare news text for LLM
    news_text = ""
    current_time = datetime.now()
    
    for i, article in enumerate(news_articles, 1):
        article_time = datetime.fromtimestamp(article["datetime"])
        minutes_ago = int((current_time - article_time).total_seconds() / 60)
        
        news_text += f"\n{i}. HEADLINE: {article['headline']}\n"
        news_text += f"   TIME: {minutes_ago} minutes ago\n"
        news_text += f"   SUMMARY: {article['summary']}\n"
    
    # Strict prompt to prevent hallucination
    prompt = f"""You are a financial news analyst. Analyze ONLY the news provided below for stock symbol: {symbol}

STRICT RULES:
1. Use ONLY the information provided in the news articles below
2. DO NOT make assumptions or add information not present in the articles
3. If no relevant news is found, return "none" for all fields
4. Focus on the MOST RECENT and MOST IMPACTFUL news item

NEWS ARTICLES:
{news_text}

Analyze and respond in EXACTLY this JSON format (no additional text):
{{
    "eventType": "<one of: earnings, order_win, upgrade, downgrade, regulatory, sector_news, merger_acquisition, product_launch, management_change, none>",
    "impact": "<one of: high, medium, low, none>",
    "direction": "<one of: positive, negative, neutral>",
    "recencyMinutes": <integer: minutes since the most relevant headline>
}}

IMPACT GUIDELINES:
- high: Major earnings beats/misses, large order wins, significant regulatory changes, M&A announcements
- medium: Analyst upgrades/downgrades, moderate contract wins, product launches
- low: General sector news, minor updates, routine announcements
- none: No relevant tradable news

DIRECTION GUIDELINES:
- positive: Good news that could drive stock price up
- negative: Bad news that could drive stock price down
- neutral: News with unclear or balanced impact

Respond with ONLY the JSON object, nothing else."""

    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MISTRAL_API_KEY}"
        }
        
        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # Low temperature to reduce hallucination
            "max_tokens": 200
        }
        
        # Use a retry/backoff wrapper to handle transient errors and rate limits
        def call_mistral_api(payload, headers, attempts=3):
            backoff = 1
            for attempt in range(1, attempts + 1):
                try:
                    resp = requests.post(MISTRAL_API_URL, headers=headers, json=payload, timeout=30)
                    if resp.status_code == 429:
                        # rate limited - backoff then retry
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    resp.raise_for_status()
                    return resp.json()
                except requests.RequestException as e:
                    if attempt == attempts:
                        raise
                    time.sleep(backoff)
                    backoff *= 2

        result = call_mistral_api(payload, headers)
        llm_response = result["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        # Remove markdown code blocks if present
        if "```json" in llm_response:
            llm_response = llm_response.split("```json")[1].split("```")[0].strip()
        elif "```" in llm_response:
            llm_response = llm_response.split("```")[1].split("```")[0].strip()
        
        analysis = json.loads(llm_response)
        
        # Validate response
        valid_event_types = ["earnings", "order_win", "upgrade", "downgrade", "regulatory", 
                            "sector_news", "merger_acquisition", "product_launch", 
                            "management_change", "none"]
        valid_impacts = ["high", "medium", "low", "none"]
        valid_directions = ["positive", "negative", "neutral"]
        
        if analysis.get("eventType") not in valid_event_types:
            analysis["eventType"] = "none"
        if analysis.get("impact") not in valid_impacts:
            analysis["impact"] = "none"
        if analysis.get("direction") not in valid_directions:
            analysis["direction"] = "neutral"
        
        # Override recencyMinutes deterministically from fetched articles
        try:
            actual_recency = min([int(a.get("minutes_ago", 9999)) for a in news_articles])
            analysis["recencyMinutes"] = int(actual_recency)
        except Exception:
            analysis.setdefault("recencyMinutes", 9999)

        return analysis
    
    except Exception as e:
        print(f"Error analyzing news with LLM for {symbol}: {str(e)}")
        return {
            "eventType": "none",
            "impact": "none",
            "direction": "neutral",
            "recencyMinutes": 9999
        }


def calculate_catalyst_score(event_type, impact, direction, recency_minutes, long_only=True):
    """
    Calculate catalyst score based on deterministic framework
    
    RecencyFactor:
      if recencyMinutes <= 60   : 1.0
      if 60 < recency <= 180    : 0.7
      if 180 < recency <= 360   : 0.4
      if recency > 360          : 0.0
    
    BaseImpactPoints:
      high   : 10
      medium :  7
      low    :  4
      none   :  0-2
    
    CatalystScore = round(BaseImpactPoints * RecencyFactor)
    Cap at 10.
    
    If long_only and direction is negative, set score to 0-2 or exclude
    """
    # Recency Factor
    if recency_minutes <= 60:
        recency_factor = 1.0
    elif recency_minutes <= 180:
        recency_factor = 0.7
    elif recency_minutes <= 360:
        recency_factor = 0.4
    else:
        recency_factor = 0.0
    
    # Base Impact Points
    impact_points = {
        "high": 10,
        "medium": 7,
        "low": 4,
        "none": 1
    }
    
    base_points = impact_points.get(impact, 1)
    
    # Calculate score
    catalyst_score = round(base_points * recency_factor)
    
    # Cap at 10
    catalyst_score = min(catalyst_score, 10)
    
    # Long-only filter: penalize negative news
    if long_only and direction == "negative":
        catalyst_score = min(catalyst_score, 2)
    
    return catalyst_score


def process_catalyst_for_stock(symbol):
    """
    Main function to process catalyst scoring for a stock
    Returns: dict with all catalyst data
    ALWAYS returns valid data - never NaN
    """
    try:
        # Fetch news from Google News RSS (no date filter needed, gets recent news)
        news_articles = fetch_google_news_rss(symbol)
        
        if not news_articles:
            print(f"   No news found for {symbol}")
            return {
                "EventType": "none",
                "Impact": "none",
                "Direction": "neutral",
                "RecencyMinutes": 9999,
                "CatalystScore_0_10": 0
            }
        
        print(f"   Found {len(news_articles)} news articles for {symbol}")
        
        # Analyze with LLM
        analysis = analyze_news_with_llm(symbol, news_articles)
        
        # Validate analysis result
        if not analysis or not isinstance(analysis, dict):
            print(f"   LLM analysis failed for {symbol}, using defaults")
            return {
                "EventType": "none",
                "Impact": "none",
                "Direction": "neutral",
                "RecencyMinutes": 9999,
                "CatalystScore_0_10": 0
            }
        
        # Calculate score
        catalyst_score = calculate_catalyst_score(
            event_type=analysis.get("eventType", "none"),
            impact=analysis.get("impact", "none"),
            direction=analysis.get("direction", "neutral"),
            recency_minutes=analysis.get("recencyMinutes", 9999),
            long_only=True
        )
        
        result = {
            "EventType": analysis.get("eventType", "none"),
            "Impact": analysis.get("impact", "none"),
            "Direction": analysis.get("direction", "neutral"),
            "RecencyMinutes": analysis.get("recencyMinutes", 9999),
            "CatalystScore_0_10": catalyst_score
        }
        
        print(f"   Result: {result['EventType']}, Impact: {result['Impact']}, Score: {result['CatalystScore_0_10']}")
        return result
        
    except Exception as e:
        # If ANY error occurs, return safe defaults
        print(f"   ERROR processing {symbol}: {str(e)}")
        return {
            "EventType": "none",
            "Impact": "none",
            "Direction": "neutral",
            "RecencyMinutes": 9999,
            "CatalystScore_0_10": 0
        }


print("\n" + "="*50)
print("STEP-4: Processing Catalyst Scores...")
print("="*50)

import time

# Reset index to ensure clean merging
df = df.reset_index(drop=True)

catalyst_results = []
success_count = 0
no_news_count = 0
error_count = 0

# Process each stock ONCE (using index to avoid duplicates)
for idx in range(len(df)):
    symbol = df.loc[idx, "Symbol"]
    print(f"\nProcessing {idx+1}/{len(df)}: {symbol}")
    
    result = process_catalyst_for_stock(symbol)
    catalyst_results.append(result)
    
    # Track statistics
    if result["CatalystScore_0_10"] > 0:
        success_count += 1
    elif result["EventType"] == "none" and result["RecencyMinutes"] == 9999:
        no_news_count += 1
    
    # Add small delay to avoid rate limiting (0.5 seconds between calls)
    if idx < len(df) - 1:  # Don't delay after last stock
        time.sleep(0.5)
    
    # Show progress every 10 stocks
    if (idx + 1) % 10 == 0:
        print(f"\n--- Progress: {idx+1}/{len(df)} stocks processed ---")
        print(f"    Stocks with scores > 0: {success_count}")
        print(f"    Stocks with no news: {no_news_count}")

# Add catalyst data to dataframe (by index, not concat)
catalyst_df = pd.DataFrame(catalyst_results)

# Ensure both dataframes have the same length
if len(catalyst_df) != len(df):
    raise ValueError(f"Mismatch: df has {len(df)} rows but catalyst_df has {len(catalyst_df)} rows!")

# Add each catalyst column to the original dataframe
for col in catalyst_df.columns:
    df[col] = catalyst_df[col].values

print("\n" + "="*50)
print("Catalyst scoring completed!")
print("="*50)
print(f"Total stocks processed: {len(df)}")
print(f"Stocks with catalyst scores > 0: {success_count}")
print(f"Stocks with no news found: {no_news_count}")
print(f"Average Catalyst Score: {df['CatalystScore_0_10'].mean():.2f}")
print(f"Max Catalyst Score: {df['CatalystScore_0_10'].max()}")
print(f"Stocks with NaN values: {df['CatalystScore_0_10'].isna().sum()}")

# ==============================
# STEP-5: TOTAL VELOCITY SCORE (0–100)
# ==============================
print("\n" + "="*50)
print("STEP-5: Calculating Total Velocity Score...")
print("="*50)

# Calculate Total Velocity Score (sum of all 4 components)
df["VelocityScore_0_100"] = (
    df["VolatilityScore_0_40"] + 
    df["VolumeShockScore_0_25"] + 
    df["RSScore_0_25"] + 
    df["CatalystScore_0_10"]
)

print(f"\nVelocity Score Statistics:")
print(f"  Mean: {df['VelocityScore_0_100'].mean():.2f}")
print(f"  Median: {df['VelocityScore_0_100'].median():.2f}")
print(f"  Min: {df['VelocityScore_0_100'].min():.2f}")
print(f"  Max: {df['VelocityScore_0_100'].max():.2f}")

# ==============================
# OPERATIONAL CONSTRAINTS
# ==============================
print("\n" + "="*50)
print("Applying Operational Constraints...")
print("="*50)

# Sort all stocks by VelocityScore (descending) - NO FILTERING
print(f"\nTotal stocks: {len(df)}")
df_sorted = df.sort_values("VelocityScore_0_100", ascending=False).reset_index(drop=True)

# Optional: Max 2 stocks per sector (if Sector column exists)
if "Sector" in df_sorted.columns:
    print(f"\nApplying max 2 stocks per sector constraint...")
    
    # Group by sector and take top 2 from each
    df_final = df_sorted.groupby("Sector").head(2).reset_index(drop=True)
    
    print(f"After max 2 per sector filter: {len(df_final)} stocks")
    
    # Show sector distribution
    sector_counts = df_final["Sector"].value_counts()
    print(f"\nSector Distribution (Top 10):")
    for sector, count in sector_counts.head(10).items():
        print(f"  {sector}: {count} stocks")
else:
    print(f"\n⚠️  No 'Sector' column found. Skipping sector-based filtering.")
    print(f"   Keeping all stocks sorted by VelocityScore.")
    
    # Keep all stocks, just sorted
    df_final = df_sorted.copy()

# ==============================
# FINAL SUMMARY
# ==============================
print("\n" + "="*50)
print("FINAL SUMMARY")
print("="*50)
print(f"Total stocks processed: {len(df)}")
print(f"Final stocks in output: {len(df_final)}")
print(f"Stocks sorted by VelocityScore (highest to lowest)")

if len(df_final) > 0:
    print(f"\nTop 10 Stocks by Velocity Score:")
    display_cols = ["Symbol", "VolatilityScore_0_40", "VolumeShockScore_0_25", 
                   "RSScore_0_25", "CatalystScore_0_10", "VelocityScore_0_100"]
    # Only show columns that exist
    display_cols = [col for col in display_cols if col in df_final.columns]
    print(df_final[display_cols].head(10).to_string(index=False))

# ==============================
# OUTPUT
# ==============================
# Save both the full dataset and the filtered dataset
OUTPUT_FILE_ALL = os.path.join(OUTPUT_DIR, "phase2_step1_2_3_4_with_catalyst_ALL.xlsx")
OUTPUT_FILE_FILTERED = os.path.join(OUTPUT_DIR, "phase2_step1_2_3_4_5_FINAL.xlsx")

df.to_excel(OUTPUT_FILE_ALL, index=False)
df_final.to_excel(OUTPUT_FILE_FILTERED, index=False)

print("\n" + "="*50)
print("Phase-2 All Steps (1-5) Completed Successfully!")
print("="*50)
print(f"Full dataset (all stocks):     {OUTPUT_FILE_ALL}")
print(f"Filtered dataset (final picks): {OUTPUT_FILE_FILTERED}")
print("="*50)

