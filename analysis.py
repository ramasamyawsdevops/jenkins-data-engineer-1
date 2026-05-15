import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directory
os.makedirs("output", exist_ok=True)

# Load dataset
df = pd.read_csv("data/orders.csv")

# -----------------------------
# Data Cleaning
# -----------------------------

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df.fillna(0, inplace=True)

# Convert date column
df["order_date"] = pd.to_datetime(df["order_date"])

# Create total amount column
df["total_amount"] = df["qty"] * df["price"]

# Save cleaned data
df.to_csv("output/cleaned_orders.csv", index=False)

print("Data Cleaning Completed")

# -----------------------------
# NumPy Operations
# -----------------------------

sales_array = np.array(df["total_amount"])

print("\nTotal Revenue:", np.sum(sales_array))
print("Average Revenue:", np.mean(sales_array))
print("Maximum Sale:", np.max(sales_array))
print("Minimum Sale:", np.min(sales_array))

# -----------------------------
# KPI Analysis
# -----------------------------

total_revenue = df["total_amount"].sum()
total_orders = df["order_id"].count()
avg_order_value = df["total_amount"].mean()

print("\n===== KPI REPORT =====")
print("Total Revenue:", total_revenue)
print("Total Orders:", total_orders)
print("Average Order Value:", round(avg_order_value, 2))

# -----------------------------
# Monthly Sales Chart
# -----------------------------

monthly_sales = (
    df.groupby(df["order_date"].dt.month)["total_amount"]
    .sum()
)

plt.figure(figsize=(8,5))
monthly_sales.plot(kind="line", marker="o")

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.tight_layout()
plt.savefig("output/monthly_sales.png")

print("monthly_sales.png saved")

# -----------------------------
# Top Products Chart
# -----------------------------

top_products = (
    df.groupby("product")["total_amount"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,5))

sns.barplot(
    x=top_products.index,
    y=top_products.values
)

plt.xticks(rotation=45)

plt.title("Top Product Sales")
plt.xlabel("Products")
plt.ylabel("Revenue")

plt.tight_layout()
plt.savefig("output/top_products.png")

print("top_products.png saved")

# -----------------------------
# Regional Sales Chart
# -----------------------------

regional_sales = (
    df.groupby("city")["total_amount"]
    .sum()
)

plt.figure(figsize=(8,5))

regional_sales.plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.ylabel("")

plt.title("Regional Sales Distribution")

plt.tight_layout()
plt.savefig("output/regional_sales.png")

print("regional_sales.png saved")

# -----------------------------
# Correlation Heatmap
# -----------------------------

numeric_df = df[["qty", "price", "total_amount"]]

plt.figure(figsize=(6,4))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("output/correlation_heatmap.png")

print("correlation_heatmap.png saved")

print("\nProject Completed Successfully")