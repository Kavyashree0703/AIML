import joblib
import numpy as np


# ==========================================
# 1. LOAD TRAINED ML MODEL
# ==========================================

model = joblib.load("chi_random_forest_model.pkl")


# ==========================================
# 2. GET UPV VALUES FROM USER
# ==========================================

print("======================================")
print("   CONCRETE HOMOGENEITY INDEX (CHI)")
print("======================================")

upv1 = float(input("Enter UPV 1 (m/s): "))
upv2 = float(input("Enter UPV 2 (m/s): "))
upv3 = float(input("Enter UPV 3 (m/s): "))


# ==========================================
# 3. MANUAL CHI CALCULATION
# ==========================================

upv_values = np.array([upv1, upv2, upv3])

mean_upv = np.mean(upv_values)

sd_upv = np.std(upv_values, ddof=1)

manual_chi = (sd_upv / mean_upv) * 100


# ==========================================
# 4. ML PREDICTION
# ==========================================

import pandas as pd

ml_input = pd.DataFrame([{
    "UPV_1 (m/s) [SYNTHETIC]": upv1,
    "UPV_2 (m/s) [SYNTHETIC]": upv2,
    "UPV_3 (m/s) [SYNTHETIC]": upv3
}])

predicted_chi = model.predict(ml_input)[0]

# ==========================================
# 5. CALCULATE DIFFERENCE
# ==========================================

difference = abs(manual_chi - predicted_chi)


# ==========================================
# 6. DISPLAY RESULTS
# ==========================================

print("\n======================================")
print("             RESULTS")
print("======================================")

print(f"Mean UPV          : {mean_upv:.4f} m/s")
print(f"Standard Deviation: {sd_upv:.4f} m/s")

print("--------------------------------------")

print(f"Manual CHI        : {manual_chi:.4f}")
print(f"ML Predicted CHI  : {predicted_chi:.4f}")

print("--------------------------------------")

print(f"Difference        : {difference:.4f}")

print("======================================")