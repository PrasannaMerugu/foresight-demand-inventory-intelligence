import pandas as pd
import joblib

sales = pd.read_csv("data/raw/sales_daily.csv")
sales["Date"] = pd.to_datetime(sales["Date"])
sales = sales.sort_values(["SKU", "Date"])

model = joblib.load("models/foresight_forecast_model.joblib")

sales["lag1"] = sales.groupby("SKU")["Units_Sold"].shift(1)
sales["lag4"] = sales.groupby("SKU")["Units_Sold"].shift(4)
sales["lag8"] = sales.groupby("SKU")["Units_Sold"].shift(8)
sales["month"] = sales["Date"].dt.month
sales["is_weekend"] = (sales["Date"].dt.dayofweek >= 5).astype(int)
sales["is_holiday"] = 0

test = sales[sales["Date"] >= "2025-12-02"].dropna().copy()

features = ["lag1", "lag4", "lag8", "month", "is_weekend", "is_holiday"]
test["RF_Pred"] = model.predict(test[features])

test["Naive_Pred"] = test.groupby("SKU")["Units_Sold"].shift(7)

test = test.dropna(subset=["Naive_Pred"])

def wape(actual, predicted):
    return (abs(actual - predicted).sum() / actual.sum()) * 100

rf_wape = wape(test["Units_Sold"], test["RF_Pred"])
naive_wape = wape(test["Units_Sold"], test["Naive_Pred"])

print(f"Test period: {test['Date'].min().date()} to {test['Date'].max().date()}")
print(f"Rows evaluated: {len(test)}")
print(f"Random Forest WAPE: {rf_wape:.2f}%")
print(f"Seasonal Naive WAPE: {naive_wape:.2f}%")
