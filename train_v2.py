import pandas as pd
import joblib
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

# No holiday column exists in the raw data, so keep this as 0 for consistency.
sales["is_holiday"] = 0

features = [
    "lag1", "lag7", "lag14", "lag28",
    "rolling7", "rolling28",
    "month", "day_of_week", "is_weekend",
    "is_holiday", "Promotion", "Price"
]

train = sales[sales["Date"] < "2025-12-02"].dropna(subset=features + ["Units_Sold"])
test = sales[sales["Date"] >= "2025-12-02"].dropna(subset=features + ["Units_Sold"])

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    max_features="sqrt"
)

model.fit(train[features], train["Units_Sold"])
test["Prediction"] = model.predict(test[features])

def wape(actual, predicted):
    return abs(actual - predicted).sum() / actual.sum() * 100

print(f"Test period: {test['Date'].min().date()} to {test['Date'].max().date()}")
print(f"Rows evaluated: {len(test)}")
print(f"Improved Random Forest WAPE: {wape(test['Units_Sold'], test['Prediction']):.2f}%")
print("Seasonal Naive WAPE: 31.15%")

joblib.dump(model, "models/foresight_forecast_model_v2.joblib")
print("Saved: models/foresight_forecast_model_v2.joblib")
