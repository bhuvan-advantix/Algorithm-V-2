import pandas as pd
import numpy as np
import os

# ======================================================
# CONFIG
# ======================================================
PHASE3_FILE = "phase-3results/Phase-3.xlsx"
OUTPUT_DIR = "phase-4results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "phase-4A_weights.xlsx")

EXPECTANCY_COL = "expectancy"   # MUST come from Phase-6

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================================
# PHASE 4A — WEIGHT DEFINITION
# ======================================================

def compute_phase4A_weights(file_path):
    df = pd.read_excel(file_path)

    if EXPECTANCY_COL not in df.columns:
        raise ValueError(
            "❌ 'expectancy' column missing.\n"
            "Phase-4A requires Phase-6 expectancy per stock."
        )

    df = df.copy()

    # 1️⃣ Clip negative expectancy
    df["s_plus"] = np.maximum(df[EXPECTANCY_COL], 0.0)

    total_s_plus = df["s_plus"].sum()

    # 2️⃣ Normalize into weights
    if total_s_plus > 0:
        df["weight"] = df["s_plus"] / total_s_plus
    else:
        df["weight"] = 0.0

    # 3️⃣ Rounding for execution safety
    df["weight"] = df["weight"].round(6)

    # 4️⃣ Validation
    print(f"✅ Phase-4A completed | Total Weight = {df['weight'].sum():.6f}")

    return df


# ======================================================
# RUN
# ======================================================
phase4A_df = compute_phase4A_weights(PHASE3_FILE)
phase4A_df.to_excel(OUTPUT_FILE, index=False)

print(f"📁 Saved → {OUTPUT_FILE}")
