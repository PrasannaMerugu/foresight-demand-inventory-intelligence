import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

sales = pd.read_csv("data/raw/sales_daily.csv")
sales["Date"] = pd.to_datetime(sales["Date"])
sales = sales.sort_values(["SKU", "Date"])

g = sales.groupby("SKU")["Units_Sold"]
sales["lag1"] = g.shift(1)
sales["lag7"] = g.shift(7)
sales["lag14"] = g.shift(14)
sales["lag28"] = g.shift(28)
sales["rolling7"] = g.transform(lambda x: x.shift(1).rolling(7).mean())
sales["rolling28"] = g.transform(lambda x: x.shift(1).rolling(28).mean())
sales["month"] = sales["Date"].dt.month
sales["day_of_week"] = sales["Date"].dt.dayofweek
sales["is_weekend"] = (sales["Date"].dt.dayofweek >= 5).astype(int)
sales["is_holiday"] = 0

features = [
    "lag1","lag7","lag14","lag28",
    "rolling7","rolling28",
    "month","day_of_week","is_weekend",
    "is_holiday","Promotion","Price"
]

sales = sales.dropna(subset=features + ["Units_Sold"]).copy()

cutoffs = [
    "2025-09-01",
    "2025-10-01",
    "2025-11-01",
    "2025-12-01"
]

results = []

for cutoff in cutoffs:
    train = sales[sales["Date"] < cutoff]
    test_end = (pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0))
    test = sales[(sales["Date"] >= cutoff) & (sales["Date"] <= test_end)]

    if len(train) == 0 or len(test) == 0:
        continue

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt"
    )

    model.fit(train[features], train["Units_Sold"])
    pred = model.predict(test[features])

    actual = test["Units_Sold"].to_numpy()

    rf_wape = np.abs(actual - pred).sum() / actual.sum() * 100

    naive = test["lag7"].to_numpy()
    naive_wape = np.abs(actual - naive).sum() / actual.sum() * 100

    results.append({
        "Cutoff": cutoff,
        "Test_Start": test["Date"].min().date(),
        "Test_End": test["Date"].max().date(),
        "Rows": len(test),
        "RF_WAPE": rf_wape,
        "Seasonal_Naive_WAPE": naive_wape
    })

result = pd.DataFrame(results)

print("\nRolling-Origin Evaluation")
print(result.round(2).to_string(index=False))
print("\nAverage RF WAPE:", round(result["RF_WAPE"].mean(), 2))
print("Average Seasonal Naive WAPE:", round(result["Seasonal_Naive_WAPE"].mean(), 2))
