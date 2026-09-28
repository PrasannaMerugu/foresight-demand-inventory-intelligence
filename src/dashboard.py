import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="FORESIGHT", layout="wide")

st.title("FORESIGHT - Demand & Inventory Intelligence")
st.caption("NorthBay Living | Inventory Risk Planning Dashboard")

BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "models" / "foresight_inventory_risk.csv")

st.subheader("Inventory Risk Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("SKUs", len(df))
c2.metric("Reorder Now", int((df["Risk"] == "Reorder Now").sum()))
c3.metric("Sales at Risk", f"\u20b9{df['Sales_at_Risk'].sum()/100000:.2f} L")
c4.metric("Locked Capital", f"\u20b9{df['Locked_Capital'].sum()/10000000:.2f} Cr")

st.subheader("Risk Distribution")
st.bar_chart(df["Risk"].value_counts())

st.subheader("SKU Risk Details")

risk_filter = st.selectbox(
    "Filter by Risk",
    ["All", "Reorder Now", "Markdown/Clear", "Watch Volatile", "Healthy"]
)

if risk_filter != "All":
    display_df = df[df["Risk"] == risk_filter]
else:
    display_df = df

st.dataframe(
    display_df[
        [
            "SKU",
            "Product_Name",
            "Category",
            "Current_Stock",
            "On_Order",
            "Lead_Time_Days",
            "Lead_Time_Demand",
            "Stock_Cover_Days",
            "Risk",
            "Sales_at_Risk",
            "Locked_Capital"
        ]
    ],
    width="stretch"
)
