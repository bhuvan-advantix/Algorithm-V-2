import pandas as pd
import numpy as np
import os
from datetime import time

# ==============================
# CONFIG
# ==============================
DAILY_FILE = "downloaded_data/daily_candles_nifty500.xlsx"
INTRADAY_PATH = "downloaded_data"
OUTPUT_FILE = "phase1_final_robust_momentum.xlsx"

MIN_TURNOVER_CR = 50
PRICE_MIN = 100
PRICE_MAX = 3000
MIN_ATR_PERCENT = 2.5
VOLUME_MULTIPLIER = 1.5
MIN_VALID_DAYS = 4   # 🔒 robustness rule

# Volatility Scoring Config
USE_PERCENTILE_SCORING = True  # True = Approach A (percentile), False = Approach B (fixed thresholds)
MAX_VOLATILITY_SCORE = 40

print("Phase-1 Started (ROBUST Momentum Gate)")

# ==============================
# LOAD DAILY DATA
# ==============================
daily_df = pd.read_excel(DAILY_FILE)
daily_df["Date"] = pd.to_datetime(daily_df["Date"])

for col in ["Open", "High", "Low", "Close", "Volume"]:
    daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

daily_df = daily_df.sort_values(["Symbol", "Date"])

# ==============================
# HELPERS
# ==============================
def calculate_daily_atr_percent(df):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    close = df["Close"].iloc[-1]
    return (atr14 / close) * 100 if close > 0 else 0.0

def calculate_vwap(df):
    vol = df["Volume"].sum()
    return (df["Close"] * df["Volume"]).sum() / vol if vol > 0 else 0.0

def calculate_volatility_score_percentile(atr_percent, all_atr_values):
    """
    Approach A: Percentile-based scoring (0-40 points)
    Calculates percentile rank within the eligible universe
    """
    if len(all_atr_values) == 0:
        return 0.0
    
    # Calculate percentile rank (0 to 1)
    percentile_rank = np.sum(all_atr_values <= atr_percent) / len(all_atr_values)
    
    # Scale to 0-40 points
    volatility_score = MAX_VOLATILITY_SCORE * percentile_rank
    return round(volatility_score, 2)

def calculate_volatility_score_fixed(atr_percent):
    """
    Approach B: Fixed threshold scoring (0-40 points)
    Uses predefined ATR% ranges
    """
    if atr_percent < 1.0:
        return 0
    elif 1.0 <= atr_percent < 1.5:
        return 10
    elif 1.5 <= atr_percent < 2.0:
        return 20
    elif 2.0 <= atr_percent < 3.0:
        return 30
    else:  # >= 3.0
        return 40


results = []

# ==============================
# PROCESS EACH STOCK
# ==============================
for symbol, sdf in daily_df.groupby("Symbol"):
    sdf = sdf.dropna()
    if len(sdf) < 20:
        continue

    # --------------------------
    # 1️⃣ LIQUIDITY GATE
    # --------------------------
    sdf["TurnoverCr"] = (sdf["Close"] * sdf["Volume"]) / 1e7
    avg_20d_turnover = sdf["TurnoverCr"].tail(20).mean()
    liquidity_pass = avg_20d_turnover >= MIN_TURNOVER_CR

    # --------------------------
    # 3️⃣ ATR% GATE
    # --------------------------
    atr_percent = calculate_daily_atr_percent(sdf)
    atr_pass = atr_percent >= MIN_ATR_PERCENT

    stock_folder = os.path.join(INTRADAY_PATH, symbol)
    if not os.path.isdir(stock_folder):
        continue

    intraday_files = sorted(os.listdir(stock_folder))

    for i, file in enumerate(intraday_files):
        trade_date = file.replace(".csv", "")
        if trade_date not in sdf["Date"].dt.strftime("%Y-%m-%d").values:
            continue

        df = pd.read_csv(os.path.join(stock_folder, file))
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Datetime"] = pd.to_datetime(trade_date + " " + df["Time"].astype(str))
        df.set_index("Datetime", inplace=True)
        df = df.sort_index()

        window = df.between_time(time(9, 15), time(9, 30))
        if window.empty:
            continue

        # --------------------------
        # 2️⃣ PRICE GATE
        # --------------------------
        cmp_price = window["Close"].iloc[-1]
        price_pass = PRICE_MIN <= cmp_price <= PRICE_MAX

        # --------------------------
        # 4️⃣ ROBUST MOMENTUM GATE
        # --------------------------
        current_volume = window["Volume"].sum()
        if current_volume <= 0:
            momentum_pass = False
            avg_5d_volume = 0
        else:
            prev_volumes = []

            for j in range(max(0, i - 5), i):
                pfile = os.path.join(stock_folder, intraday_files[j])
                pdf = pd.read_csv(pfile)

                for col in ["Open", "High", "Low", "Close", "Volume"]:
                    pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

                pdf["Datetime"] = pd.to_datetime(
                    intraday_files[j].replace(".csv", "") + " " + pdf["Time"].astype(str)
                )
                pdf.set_index("Datetime", inplace=True)

                pw = pdf.between_time(time(9, 15), time(9, 30))
                vol = pw["Volume"].sum() if not pw.empty else 0

                if vol > 0:
                    prev_volumes.append(vol)

            if len(prev_volumes) < MIN_VALID_DAYS:
                momentum_pass = False
                avg_5d_volume = 0
            else:
                avg_5d_volume = sum(prev_volumes) / len(prev_volumes)
                vwap = calculate_vwap(window)
                momentum_pass = (
                    current_volume >= avg_5d_volume * VOLUME_MULTIPLIER
                    and cmp_price > vwap
                )

        final_pass = liquidity_pass and price_pass and atr_pass and momentum_pass

        results.append({
            "Date": trade_date,
            "Symbol": symbol,
            "20D Avg Turnover ₹Cr": round(avg_20d_turnover, 2),
            "CMP ₹": round(cmp_price, 2),
            "Daily ATR %": round(atr_percent, 2),
            "Current Slot Volume": int(current_volume),
            "5D Slot Avg Volume": int(avg_5d_volume),
            "Liquidity Pass": "YES" if liquidity_pass else "NO",
            "Price Pass": "YES" if price_pass else "NO",
            "ATR Pass": "YES" if atr_pass else "NO",
            "Momentum Pass": "YES" if momentum_pass else "NO",
            "Phase-1 Final Pass": "YES" if final_pass else "NO"
        })

# ==============================
# OUTPUT
# ==============================
pd.DataFrame(results).to_excel(OUTPUT_FILE, index=False)
print(f"Phase-1 Completed → {OUTPUT_FILE}")
