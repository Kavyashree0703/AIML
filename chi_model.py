import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_excel("CHI_demo_dataset_final_no_chi.xls")


# ==========================================
# 2. SELECT UPV COLUMNS
# ==========================================

upv_columns = [
    "UPV_1 (m/s) [SYNTHETIC]",
    "UPV_2 (m/s) [SYNTHETIC]",
    "UPV_3 (m/s) [SYNTHETIC]"
]


# ==========================================
# 3. CALCULATE MEAN UPV
# ==========================================

df["UPV_Mean"] = df[upv_columns].mean(axis=1)


# ==========================================
# 4. CALCULATE STANDARD DEVIATION
# ==========================================

df["UPV_SD"] = df[upv_columns].std(axis=1)


# ==========================================
# 5. CALCULATE CHI
# ==========================================

df["CHI"] = (df["UPV_SD"] / df["UPV_Mean"]) * 100


# ==========================================
# 6. CREATE INPUT (X) AND TARGET (y)
# ==========================================

X = df[upv_columns]

y = df["CHI"]


# ==========================================
# 7. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ==========================================
# 8. CREATE RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# ==========================================
# 9. TRAIN MODEL
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 10. PREDICT CHI
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 11. DISPLAY ACTUAL VS PREDICTED
# ==========================================

results = pd.DataFrame({
    "Actual_CHI": y_test.values,
    "Predicted_CHI": y_pred
})

print("\nActual vs Predicted CHI:\n")
print(results.head(10))
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred) ** 0.5

# Calculate R²
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE  :", mae)
print("RMSE :", rmse)
print("R²   :", r2)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.5)

# Ideal prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.xlabel("Actual CHI")
plt.ylabel("Predicted CHI")
plt.title("Actual vs Predicted CHI")

plt.grid(True)
plt.show()
import joblib

joblib.dump(model, "chi_random_forest_model.pkl")