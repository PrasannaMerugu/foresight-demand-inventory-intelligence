import pandas as pd
from pathlib import Path

# Project root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Raw data folder
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Load datasets
sales = pd.read_csv(RAW_DATA_DIR / "sales_daily.csv")
sku_master = pd.read_csv(RAW_DATA_DIR / "sku_master.csv")
calendar = pd.read_csv(RAW_DATA_DIR / "calendar.csv")
inventory = pd.read_csv(RAW_DATA_DIR / "inventory_snapshots.csv")

# Display basic information
print("Sales Daily:")
print(sales.head())

print("\nSKU Master:")
print(sku_master.head())

print("\nCalendar:")
print(calendar.head())

print("\nInventory Snapshots:")
print(inventory.head())