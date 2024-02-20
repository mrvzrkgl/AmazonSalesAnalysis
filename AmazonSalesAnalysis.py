# Amazon Sale Report Analysis

# importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# importing dataset
amazon = pd.read_csv("C:/Users/zrkgl/Desktop/AmazonSalesAnalysis/AmazonSaleReport.csv")
amazon.head()


# Variables

"""
OrderID: Order number or unique identifier.
Date: Order date or transaction date.
Status: Order status (e.g., Shipped, Delivered, Pending, etc.).
Fulfilment: How the order was fulfilled (e.g., FBA, FBM, etc.).
SalesChannel: Sales channel (e.g., Amazon, eBay, etc.).
ship-service-level: Selected shipping service level (e.g., Standard, Expedited, etc.).
Style: Style or model of the product.
SKU: Stock code or unique identifier of the product.
Category: Category of the product (e.g., Electronics, Clothing, Books, etc.).
Size: Size of the product (e.g., Small, Medium, Large, etc.).
ASIN: Amazon Standard Identification Number, a code used to identify products on Amazon.
CourierStatus: Courier status (e.g., In Transit, Delivered, Pending, etc.).
Qty: Quantity of the product sold.
currency: Currency in which the transaction was made (e.g., USD, EUR, GBP, etc.).
Amount: Transaction amount.
ship-city: City of the delivery address.
ship-state: State or region of the delivery address.
ship-postal-code: Postal code of the delivery address.
ship-country: Country of the delivery address.
promotion-ids: Identifiers of the applied promotions (e.g., Promo1, Promo2, Promo3, etc.).
B2B: A logical value indicating whether it is a business-to-business (B2B) sale (True or False).
fulfilled-by: A value indicating who fulfilled the order (e.g., Seller, Amazon, etc.).
"""

amazon = amazon.rename(columns=lambda x: x.strip().replace(" ", "") if isinstance(x, str) else x)

# change type
amazon = pd.DataFrame(amazon)

# remove index
amazon.drop(columns=['index'], inplace=True)

# missing value check
amazon.isnull().sum()


# missing
amazon['Amount'] = amazon['Amount'].interpolate(method='linear')

amazon['CourierStatus'] = amazon['CourierStatus'].fillna(amazon['CourierStatus'].mode()[0])
amazon['currency'] = amazon['currency'].fillna(amazon['currency'].mode()[0])
amazon['ship-city'] = amazon['ship-city'].fillna(amazon['ship-city'].mode()[0])
amazon['ship-state'] = amazon['ship-state'].fillna(amazon['ship-state'].mode()[0])
amazon['ship-postal-code'] = amazon['ship-postal-code'].fillna(amazon['ship-postal-code'].mode()[0])
amazon['ship-country'] = amazon['ship-country'].fillna(amazon['ship-country'].mode()[0])
amazon['promotion-ids'] = amazon['promotion-ids'].fillna(amazon['promotion-ids'].mode()[0])
amazon['fulfilled-by'] = amazon['fulfilled-by'].fillna(amazon['fulfilled-by'].mode()[0])
amazon['Unnamed:22'] = amazon['Unnamed:22'].fillna(amazon['Unnamed:22'].mode()[0])



## Numpy

amazon.shape   # 23 columns, 128975 rows
amazon.size
amazon.dtypes


# Descriptive Statistics

# mean
mean_values = {col: np.mean(amazon[col]) for col in amazon.columns if amazon[col].dtype in ['int64', 'float64']}

for col, mean in mean_values.items():
    print(f"Mean {col}: {mean}")


# median
median_values = {col: np.median(amazon[col]) for col in amazon.columns if amazon[col].dtype in ['int64', 'float64']}

for col, median in median_values.items():
    print(f"Median {col}: {median}")


# variance
variance_values = {col: np.var(amazon[col]) for col in amazon.columns if amazon[col].dtype in ['int64', 'float64']}

for col, var in variance_values.items():
    print(f"Variance {col}: {var}")


# standard deviation
std_values = {col: np.std(amazon[col]) for col in amazon.columns if amazon[col].dtype in ['int64', 'float64']}

for col, std in std_values.items():
    print(f"Variance {col}: {std}")




## Pandas

amazon["Status"].value_counts()
amazon["Fulfilment"].value_counts()
amazon["SalesChannel"].value_counts()
amazon["ship-service-level"].value_counts()
amazon["Style"].value_counts()
amazon["SKU"].value_counts()
amazon["Category"].value_counts()
amazon["Size"].value_counts()
amazon["ASIN"].value_counts()
amazon["CourierStatus"].value_counts()
amazon["ship-city"].value_counts()
amazon["ship-state"].value_counts()
amazon["ship-country"].value_counts()
amazon["B2B"].value_counts()
amazon["fulfilled-by"].value_counts()
amazon["Unnamed:22"].value_counts()



# Categorical Variables

amazon["Fulfilment"].value_counts().plot(kind="bar")
plt.show()

amazon["Status"].value_counts().plot(kind="bar")
plt.show()

amazon["SalesChannel"].value_counts().plot(kind="bar")
plt.show()

amazon["ship-service-level"].value_counts().plot(kind="bar")
plt.show()

amazon["Category"].value_counts().plot(kind="bar")
plt.show()

amazon["Size"].value_counts().plot(kind="bar")
plt.show()

amazon["CourierStatus"].value_counts().plot(kind="bar")
plt.show()

amazon["currency"].value_counts().plot(kind="bar")
plt.show()

amazon["fulfilled-by"].value_counts().plot(kind="bar")
plt.show()

amazon["B2B"].value_counts().plot(kind="bar")
plt.show()


# Numerical Variables

# Histograms
plt.hist(amazon["Qty"])
plt.hist(amazon["Amount"])
plt.hist(amazon["ship-postal-code"])

# Boxplots
plt.boxplot(amazon["Qty"])
plt.boxplot(amazon["Amount"])
plt.boxplot(amazon["ship-postal-code"])

# There are a lot of outliers especially in the Qty and Amount variables



## Matplotlib

# Line Graph

# Trend of sales by weeks
amazon['Date'] = pd.to_datetime(amazon['Date'], format='%m-%d-%y')
weekly_sales = amazon.groupby(amazon['Date'].dt.isocalendar().week)['Qty'].sum()

# Plot weekly sales
plt.figure(figsize=(10,6))
plt.plot(weekly_sales.index, weekly_sales.values)
plt.xlabel('Weeks')
plt.ylabel('Total Qty')
plt.title('Weekly Qty')
plt.grid(True)
plt.show()


# Bar Chart

# Bar chart for total QTY by Category
plt.figure(figsize=(12, 6))

qty_by_category = amazon.groupby('Category')['Qty'].sum()
plt.bar(qty_by_category.index, qty_by_category.values)

plt.xlabel('Category')
plt.ylabel('Total Quantity')
plt.title('Total Quantity by Category')
plt.show()


# Bar chart for total Amount by Category
plt.figure(figsize=(12, 6))
qty_by_category = amazon.groupby('Category')['Amount'].sum()
plt.bar(qty_by_category.index, qty_by_category.values)

plt.xlabel('Category')
plt.ylabel('Total Amount')
plt.title('Total Amount by Category')
plt.show()


## Seaborn

# Scatter Plot
plt.figure(figsize=(12, 6))
sns.scatterplot(data = amazon, x = "Qty", y = "Amount")

plt.xlabel("Qty")
plt.ylabel("Amount")
plt.title("Qty and Amount's Scatter Plot")


# Regression Model
from sklearn.linear_model import LinearRegression

X = amazon[['Qty']]
y = amazon['Amount']
model = LinearRegression()
model.fit(X, y)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Qty')
plt.ylabel('Amount')
plt.title('Relationship between Qty and Amount')
plt.legend()
plt.show()



## Cases


# The Most Popular Categories
popular_categories = category_sales.nlargest(5, 'Qty')
print("Most Popular Categories:\n", popular_categories)

plt.figure(figsize=(12, 6))
sns.barplot(x='Qty', y='Category', data=popular_categories, palette='viridis')
plt.title('Most Popular Categories')
plt.xlabel('Quantity Sold')
plt.ylabel('Category')
plt.show()



# Time Dependent Analysis (monthly)
amazon['Date'] = pd.to_datetime(amazon['Date'])
amazon['Month'] = amazon['Date'].dt.month
monthly_sales = amazon.groupby(['Category', 'Month']).agg({'Qty': 'sum'}).reset_index()
print("Monthly Sales:\n", monthly_sales)

plt.figure(figsize=(12, 14))
sns.lineplot(x='Month', y='Qty', hue='Category', data=monthly_sales, palette='tab10', marker='o')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Quantity Sold')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()



# Monthly Sales Trend
plt.figure(figsize=(12, 8))
monthly_total_sales = amazon.groupby('Month')['Qty'].sum()
sns.lineplot(x=monthly_total_sales.index, y=monthly_total_sales.values, marker='o', color='blue')
plt.title('Monthly Total Sales')
plt.xlabel('Month')
plt.ylabel('Total Quantity Sold')
plt.xticks(range(1, 13))
plt.grid(True)
plt.show()




## Customer Segmentation with RFM Analysis

from datetime import datetime

amazon['Date'] = pd.to_datetime(amazon['Date'], format='%m-%d-%y')

today = pd.to_datetime('today')
amazon.head()

recency = today - amazon.groupby('OrderID')['Date'].max()
frequency = amazon.groupby('OrderID').size()
monetary = amazon.groupby('OrderID')['Amount'].sum()

rfm_table = pd.DataFrame({
    'Recency': recency,
    'Frequency': frequency,
    'Monetary': monetary
})

rfm_table.head()

rfm_table['Recency_Quartile'] = pd.qcut(rfm_table['Recency'], q=5, labels=False, duplicates='drop') + 1
freq_quartiles = pd.qcut(rfm_table['Frequency'], q=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=False, duplicates='drop') + 1
monetary_quartiles = pd.qcut(rfm_table['Monetary'], q=[0, 0.2, 0.4, 0.6, 0.8, 1], labels=False, duplicates='drop') + 1

rfm_table['RFM_Score'] = rfm_table['Recency_Quartile'].astype(str) + freq_quartiles.astype(str) + monetary_quartiles.astype(str)
rfm_table.head()

rfm_summary = rfm_table.groupby('RFM_Score').agg({
    'Recency': ['mean', 'min', 'max', 'count'],
    'Frequency': ['mean', 'min', 'max', 'count'],
    'Monetary': ['mean', 'min', 'max', 'count']
}).round(1)

pd.set_option('display.max_columns', None)
rfm_summary


"""
Customer segmentation was made with the analysis. If the segmentation number 114 is interpreted:
There were 5414 orders in this segment.
On average, the last shopping was ordered 601 days ago.
Shopping frequency is 1.
"""


