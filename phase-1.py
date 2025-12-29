import pandas as pd
import numpy as np
import os
from datetime import time

# ==============================
# CONFIG
# ==============================
DAILY_FILE = "downloaded_data/daily_candles_nifty500.xlsx"
INTRADAY_PATH = "downloaded_data"
OUTPUT_FILE = "phase-1results/phase1_final_robust_momentum.xlsx"

MIN_TURNOVER_CR = 50
PRICE_MIN = 100
PRICE_MAX = 3000
MIN_ATR_PERCENT = 2.5
VOLUME_MULTIPLIER = 1.5
MIN_VALID_DAYS = 4   # 🔒 robustness rule
MAX_SPREAD_PERCENT = 0.5  # percent (max allowed bid-ask / high-low spread relative to CMP)
# Time window configuration (default kept at 9:15-9:30)
# To test an alternate window, set USE_ALTERNATE_TIME = True
TIME_START = time(9, 15)
TIME_END = time(9, 30)
USE_ALTERNATE_TIME = False
ALT_TIME_START = time(9, 15)
ALT_TIME_END = time(9, 45)

# Diagnostic cross-check (prints gate-level counts to console only)
DIAGNOSTIC = True

print("Phase-1 Started (ROBUST Momentum Gate)")

# -- Validate input paths early to fail fast
if not os.path.exists(DAILY_FILE):
    raise FileNotFoundError(f"Daily data not found: {DAILY_FILE}")
if not os.path.exists(INTRADAY_PATH):
    raise FileNotFoundError(f"Intraday path not found: {INTRADAY_PATH}")
# Ensure output directory exists
out_dir = os.path.dirname(OUTPUT_FILE)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

# ==============================
# LOAD DAILY DATA
# ==============================
daily_df = pd.read_excel(DAILY_FILE)
daily_df["Datetime"] = pd.to_datetime(daily_df["Datetime"])

for col in ["Open", "High", "Low", "Close", "Volume"]:
    daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

daily_df = daily_df.sort_values(["Symbol", "Datetime"])

# ==============================
# HELPERS
# ==============================
def calculate_daily_atr_percent_raw(df):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    close = df["Close"].iloc[-1]
    return (atr14 / close) * 100 if close > 0 else 0.0

def calculate_vwap_typical(df):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].sum()
    return (typical_price * df["Volume"]).sum() / vol if vol > 0 else 0.0

# ==============================
# PROCESS EACH STOCK
# ==============================
results = []
intraday_cache = {}

# Diagnostic counters / samples (console-only)
diag = {
    'total_symbols': 0,
    'no_intraday_folder': [],
    'too_few_daily_rows': [],
    'liquidity_failed': [],
    'atr_failed': [],
    'no_intraday_date_match': [],
    'sanity_failed': [],
    'price_failed': [],
    'momentum_failed': [],
    'spread_failed': [],
    'final_passed': []
}

for symbol, sdf in daily_df.groupby("Symbol"):
    sdf = sdf.dropna()
    diag['total_symbols'] += 1
    if len(sdf) < 20:
        diag['too_few_daily_rows'].append(symbol)
        continue

    # 1️⃣ LIQUIDITY GATE
    sdf["TurnoverCr"] = (sdf["Close"] * sdf["Volume"]) / 1e7
    avg_20d_turnover = sdf["TurnoverCr"].tail(20).mean()
    liquidity_pass = avg_20d_turnover >= MIN_TURNOVER_CR
    if not liquidity_pass:
        diag['liquidity_failed'].append(symbol)

    # 2️⃣ ATR GATE (RAW)
    atr_percent_raw = calculate_daily_atr_percent_raw(sdf)
    atr_pass = atr_percent_raw >= MIN_ATR_PERCENT
    if not atr_pass:
        diag['atr_failed'].append(symbol)

    stock_folder = os.path.join(INTRADAY_PATH, symbol)
    if not os.path.isdir(stock_folder):
        diag['no_intraday_folder'].append(symbol)
        continue

    intraday_files = sorted(os.listdir(stock_folder))

    for i, file in enumerate(intraday_files):
        trade_date = file.replace(".csv", "")
        if trade_date not in sdf["Datetime"].dt.strftime("%Y-%m-%d").values:
            # intraday date doesn't match daily data for this symbol
            diag['no_intraday_date_match'].append((symbol, trade_date))
            continue
        file_path = os.path.join(stock_folder, file)
        if file_path not in intraday_cache:
            intraday_cache[file_path] = pd.read_csv(file_path)
        df = intraday_cache[file_path].copy()
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Datetime"] = pd.to_datetime(trade_date + " " + df["Time"].astype(str))
        df.set_index("Datetime", inplace=True)
        # remove duplicate timestamps if present
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]
        df = df.sort_index()

        # choose time window (respects USE_ALTERNATE_TIME)
        ts = ALT_TIME_START if USE_ALTERNATE_TIME else TIME_START
        te = ALT_TIME_END if USE_ALTERNATE_TIME else TIME_END
        window = df.between_time(ts, te)
        if window.empty:
            continue

        # basic sanity checks (skip only this slot if invalid)
        if window["Volume"].sum() == 0 or window["Close"].iloc[-1] <= 0:
            diag['sanity_failed'].append((symbol, trade_date))
            continue

        # 3️⃣ PRICE GATE
        cmp_price = window["Close"].iloc[-1]
        if cmp_price <= 0:
            continue
        price_pass = PRICE_MIN <= cmp_price <= PRICE_MAX
        if not price_pass:
            diag['price_failed'].append((symbol, trade_date, cmp_price))

        # Spread check (High-Low in the slot relative to CMP)
        spread_pct = ((window["High"].iloc[-1] - window["Low"].iloc[-1]) / cmp_price) * 100
        spread_pass = spread_pct <= MAX_SPREAD_PERCENT
        if not spread_pass:
            diag['spread_failed'].append((symbol, trade_date, round(spread_pct,4)))

        # 4️⃣ ROBUST MOMENTUM GATE
        current_volume = window["Volume"].sum()
        prev_volumes = []
        # explicit lookback over up to 5 prior files (previous trading days)
        lookback_start = max(0, i - 5)
        for j in range(lookback_start, i):
            prev_path = os.path.join(stock_folder, intraday_files[j])
            if prev_path not in intraday_cache:
                intraday_cache[prev_path] = pd.read_csv(prev_path)
            pdf = intraday_cache[prev_path].copy()
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

            pdf["Datetime"] = pd.to_datetime(
                intraday_files[j].replace(".csv", "") + " " + pdf["Time"].astype(str)
            )
            pdf.set_index("Datetime", inplace=True)

            # remove duplicate timestamps if present in historical intraday
            if pdf.index.duplicated().any():
                pdf = pdf[~pdf.index.duplicated(keep='first')]

            pw = pdf.between_time(ts, te)
            vol = pw["Volume"].sum() if not pw.empty else 0
            if vol > 0:
                prev_volumes.append(vol)

        if len(prev_volumes) < MIN_VALID_DAYS:
            momentum_pass = False
            avg_5d_volume = 0
            vol_mult = 0
            vwap = 0
            above_vwap = "NO"
        else:
            avg_5d_volume = np.mean(prev_volumes)
            vol_mult = current_volume / avg_5d_volume if avg_5d_volume > 0 else 0
            vwap = calculate_vwap_typical(window)
            above_vwap = "YES" if cmp_price > vwap else "NO"

            momentum_pass = (
                current_volume >= avg_5d_volume * VOLUME_MULTIPLIER
                and cmp_price > vwap
            )
            if not momentum_pass:
                diag['momentum_failed'].append((symbol, trade_date))

        final_pass = liquidity_pass and price_pass and atr_pass and momentum_pass and spread_pass
        if final_pass:
            diag['final_passed'].append((symbol, trade_date))

        results.append({
            "Date": trade_date,
            "Symbol": symbol,

            "20D Avg Turnover ₹Cr": round(avg_20d_turnover, 4),
            "CMP ₹": round(cmp_price, 4),

            "ATR% Raw": round(atr_percent_raw, 6),
            "ATR% Rounded": round(atr_percent_raw, 2),

            "Current Slot Volume": int(current_volume),
            "5D Slot Avg Volume": round(avg_5d_volume, 2),
            "VolMult": round(vol_mult, 4),

            "VWAP": round(vwap, 4),
            "Above VWAP": above_vwap,
            "Spread %": round(spread_pct, 4),
            "Spread Pass": "YES" if spread_pass else "NO",

            "Liquidity Pass": "YES" if liquidity_pass else "NO",
            "Price Pass": "YES" if price_pass else "NO",
            "ATR Pass": "YES" if atr_pass else "NO",
            "Momentum Pass": "YES" if momentum_pass else "NO",
            "Phase-1 Final Pass": "YES" if final_pass else "NO"
        })

# ==============================
# OUTPUT
# ==============================
out_df = pd.DataFrame(results)
try:
    out_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Phase-1 Completed → {OUTPUT_FILE}")
except PermissionError:
    # Fallback: write CSV if the Excel file is locked / not writable
    alt = OUTPUT_FILE.replace('.xlsx', '.csv')
    out_df.to_csv(alt, index=False)
    print(f"Phase-1 Completed → {alt} (fallback, original XLSX not writable)")

if DIAGNOSTIC:
    def sample(lst, n=5):
        try:
            return lst[:n]
        except Exception:
            return []

    print("\n=== Diagnostic Summary ===")
    print(f"Total symbols in daily feed: {diag['total_symbols']}")
    print(f"Final passed count: {len(diag['final_passed'])}")
    print(f"Symbols missing intraday folder: {len(diag['no_intraday_folder'])} -> samples: {sample(diag['no_intraday_folder'])}")
    print(f"Too few daily rows (<20): {len(diag['too_few_daily_rows'])} -> samples: {sample(diag['too_few_daily_rows'])}")
    print(f"Liquidity failed: {len(diag['liquidity_failed'])} -> samples: {sample(diag['liquidity_failed'])}")
    print(f"ATR failed: {len(diag['atr_failed'])} -> samples: {sample(diag['atr_failed'])}")
    print(f"Sanity failures (zero vol / non-positive close): {len(diag['sanity_failed'])} -> samples: {sample(diag['sanity_failed'])}")
    print(f"Price failures: {len(diag['price_failed'])} -> samples: {sample(diag['price_failed'])}")
    print(f"Momentum failures (slot): {len(diag['momentum_failed'])} -> samples: {sample(diag['momentum_failed'])}")
    print(f"Spread failures: {len(diag['spread_failed'])} -> samples: {sample(diag['spread_failed'])}")
    print(f"Intraday date mismatches logged: {len(diag['no_intraday_date_match'])} -> samples: {sample(diag['no_intraday_date_match'])}")
    print("===========================\n")
