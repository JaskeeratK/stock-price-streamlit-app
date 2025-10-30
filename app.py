import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# App Title
st.set_page_config(page_title="Stock Price Prediction App", layout="wide")
st.title("Stock Price Prediction App")

# Sidebar — Company Selection
st.sidebar.header("User Input")

companies = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBI": "SBIN.NS",
    "Adani Enterprises": "ADANIENT.NS"
}

company_name = st.sidebar.selectbox("Select a company", ["-- Select a Company --"] + list(companies.keys()))
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Comparison Stocks
available_stocks = [
    "AAPL", "MSFT", "TSLA", "GOOG", "AMZN", "META",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"
]
symbols = st.sidebar.multiselect(
    "Compare with other stocks (optional):",
    available_stocks,
    default=[]
)

# Default message before selection
if company_name == "-- Select a Company --":
    st.markdown("### Let's predict some prices!")
    st.info("Select a company and date range from the sidebar to start analysis.")
    st.stop()

ticker = companies[company_name]

# Fetch Main Data
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
if data.empty:
    st.error("No data found. Please check the stock symbol or date range.")
    st.stop()

st.subheader(f"Stock Data for {company_name} ({ticker})")
st.dataframe(data.tail())

# Plot — Closing Price
st.subheader("Closing Price Over Time")
fig, ax = plt.subplots()
ax.plot(data["Close"], label=f"{company_name} Closing Price", color='blue')
ax.set_xlabel("Date")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)

# Compare Selected Stocks
if symbols:
    st.subheader("Compare Stock Performance")
    combined_df = pd.DataFrame()
    for sym in symbols:
        df = yf.download(sym, start=start_date, end=end_date, auto_adjust=True)
        if not df.empty:
            combined_df[sym] = df["Close"]
    if not combined_df.empty:
        st.line_chart(combined_df)
    else:
        st.warning("No valid data for the selected comparison stocks.")

# Prepare Data for Prediction
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(np.array(data['Close']).reshape(-1, 1))

train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for RandomForest
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

# Train Model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse Transform
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
actual_prices = scaler.inverse_transform(scaled_data)

# Align test predictions for RMSE
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
test_predict_aligned = test_predict[:len(y_test_actual)]
rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict_aligned))

# Sidebar Metrics
st.sidebar.subheader(" Model Performance")
st.sidebar.metric("RMSE", f"{rmse:.2f}")

# Plot Predictions (Date-Aligned)
st.subheader("📊 Predicted vs Actual Prices")
look_back = time_step
train_dates = data.index[look_back:look_back + len(train_predict)]
test_dates = data.index[len(train_predict) + (look_back * 2) + 1:len(scaled_data) - 1]

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(data.index, actual_prices, label='Actual Price', color='gray')
ax2.plot(train_dates, train_predict, label='Train Prediction', color='green')
ax2.plot(test_dates, test_predict, label='Test Prediction', color='orange')

ax2.set_title(f"Predicted vs Actual Prices for {company_name}")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (INR)")
ax2.legend()
st.pyplot(fig2)

# 🔮 Forecast Next 30 Days (Fixed)
st.subheader("🔮 Next 30 Days Forecast")

last_60 = scaled_data[-time_step:].flatten().reshape(1, -1)
future_predictions = []

for _ in range(30):
    next_pred = model.predict(last_60)[0]
    # add slight random variation to simulate realistic trend
    next_pred += np.random.normal(0, 0.002)
    future_predictions.append(next_pred)
    last_60 = np.append(last_60[:, 1:], next_pred).reshape(1, -1)

# Smooth and inverse-transform predictions
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_prices = pd.Series(future_prices.flatten()).rolling(3, min_periods=1).mean().values

# Create future dates and DataFrame
future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=30)
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_prices})

st.dataframe(future_df.style.format({'Predicted Price': '{:.2f}'}))

# Plot forecast
fig3, ax3 = plt.subplots(figsize=(10, 5))
ax3.plot(data.index, data['Close'], label='Historical', color='blue')
ax3.plot(future_df['Date'], future_df['Predicted Price'],
         label='Future Forecast (Next 30 Days)', color='red', linestyle='--', marker='o', markersize=3)
ax3.set_xlabel("Date")
ax3.set_ylabel("Price (INR)")
ax3.legend()
st.pyplot(fig3)

