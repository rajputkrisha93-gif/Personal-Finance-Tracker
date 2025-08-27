#!/usr/bin/env python
# coding: utf-8

# In[40]:


#Dummy Dataset 
df = pd.read_csv(r"C:\Users\Dell\Downloads\Major assignment dataset.csv")
print("First 5 rows of your data:")
print(df.head())


# In[41]:


df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")


# In[42]:


import pandas as pd  
df = pd.read_csv(r"C:\Users\Dell\Downloads\Major assignment dataset.csv")
df["Amount"] = df["Amount"].replace({",": ""}, regex=True).astype(float)

total_income = df[df["Type"] == "Income"]["Amount"].sum()
total_expense = df[df["Type"] == "Expense"]["Amount"].sum()
balance = total_income - total_expense

print(f"Total Income: {total_income}")
print(f"Total Expense: {total_expense}")
print(f"Balance (Savings): {balance}")


# In[43]:


print(df.head())  
print(df.info()) 
print(df.describe())  


# In[44]:


income_by_category = df[df["Type"] == "Income"].groupby("Category")["Amount"].sum()
expense_by_category = df[df["Type"] == "Expense"].groupby("Category")["Amount"].sum()

print("Income by Category:\n", income_by_category)
print("Expense by Category:\n", expense_by_category)


# In[45]:


import matplotlib.pyplot as plt

expense_by_category.plot(kind="bar", color="red", title="Expenses by Category")
plt.show()

income_by_category.plot(kind="bar", color="green", title="Income by Category")
plt.show()


# In[46]:


df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Grouping them by month and sum amounts
monthly_summary = df.groupby(df["Date"].dt.to_period("M"))["Amount"].sum()

print(monthly_summary)


# In[47]:


savings_rate = (balance / total_income) * 100
print(f"Savings Rate: {savings_rate:.2f}%")

print("Highest Expense Category:", expense_by_category.idxmax())
print("Highest Income Source:", income_by_category.idxmax())


# In[48]:


#Kaggle dataset 
df_kaggle = pd.read_csv(r'C:\Users\Dell\Downloads\data.csv')
print(df_kaggle.head())


# In[49]:


import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\Dell\Downloads\data.csv")

# Expense columns
expense_cols = ["Rent", "Loan_Repayment", "Insurance", "Groceries", "Transport",
                "Eating_Out", "Entertainment", "Utilities", "Healthcare",
                "Education", "Miscellaneous"]

# Total calculations
total_income = df["Income"].sum()
total_expense = df[expense_cols].sum().sum()
balance = total_income - total_expense

print(f"Total Income: ₹{total_income:.2f}")
print(f"Total Expense: ₹{total_expense:.2f}")
print(f"Balance (Savings): ₹{balance:.2f}")


# In[50]:


expense_by_category = df[expense_cols].sum()
print("\nExpense by Category:\n", expense_by_category)

# Bar chart
expense_by_category.plot(kind="bar", color="red", title="Expenses by Category")
plt.ylabel("Amount (₹)")
plt.show()


# In[51]:


potential_savings_cols = [col for col in df.columns if col.startswith("Potential_Savings_")]
potential_savings = df[potential_savings_cols].sum()
print("\nPotential Savings by Category:\n", potential_savings)

# Bar chart
potential_savings.plot(kind="bar", color="green", title="Potential Savings by Category")
plt.ylabel("Amount (₹)")
plt.show()


# In[52]:


avg_savings_rate = (df["Desired_Savings"].sum() / total_income) * 100
print(f"Average Desired Savings Rate: {avg_savings_rate:.2f}%")


# In[53]:


import yfinance as yf
import matplotlib.pyplot as plt

portfolio = {
    "INFY.NS": 10,  # Infosys
    "TCS.NS": 5     # TCS
}

total_portfolio_value = 0
labels = []
sizes = []

for ticker, qty in portfolio.items():
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")["Close"].iloc[-1]
    value = price * qty
    total_portfolio_value += value
    labels.append(ticker)
    sizes.append(value)
    print(f"{ticker}: {qty} shares x ₹{price:.2f} = ₹{value:.2f}")

print(f"\nTotal Portfolio Value: ₹{total_portfolio_value:.2f}")

# Pie chart 
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title("Portfolio Allocation")
plt.show()



# In[54]:


import numpy as np
symbol = "INFY.NS"
hist = yf.Ticker(symbol).history(period="2y")[["Close"]].dropna()
hist = hist.rename(columns={"Close":"close"})

# Train-test split (80% train, 20% test)
n = len(hist)
split = int(n * 0.8)
train = hist.iloc[:split]
test = hist.iloc[split:]

# Naive prediction: tomorrow = last train price
naive_pred = pd.Series(train["close"].iloc[-1], index=test.index)

# Moving Average prediction
window = 10
sma_value = train["close"].tail(window).mean()
sma_pred = pd.Series(sma_value, index=test.index)

# Error metrics
def mae(y, yhat): 
    return float(np.mean(np.abs(y - yhat)))

def mape(y, yhat): 
    return float(np.mean(np.abs((y - yhat) / y))) * 100

y = test["close"]
print("\n=== Stock Price Prediction (", symbol, ") ===")
print("NAIVE  -> MAE:", mae(y, naive_pred), "MAPE:", mape(y, naive_pred))
print("SMA(10)-> MAE:", mae(y, sma_pred),  "MAPE:", mape(y, sma_pred))


# In[ ]:




