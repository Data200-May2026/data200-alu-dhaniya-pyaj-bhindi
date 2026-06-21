import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path

# -----------------------------
# Load Dataset
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "companies" / "NEPSE_Cleaned.csv"

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

# -----------------------------
# Create Target Variable
# -----------------------------
df = df.sort_values(["Company", "Date"])

df["Next_Close"] = df.groupby("Company")["Close"].shift(-1)

df = df.dropna(subset=["Next_Close"])

df["Movement"] = (df["Next_Close"] > df["Close"]).astype(int)

# -----------------------------
# Features
# -----------------------------
features = [
    "Open",
    "High",
    "Low",
    "Volume",
    "Daily_Return",
    "RSI_14",
    "MACD",
    "SMA_5",
    "SMA_20"
]

X = df[features]
y = df["Movement"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_scaled, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📈 NEPSE Stock Movement Prediction")

st.write(
    "Select a company and trading date to predict whether the next day's stock price will increase or decrease."
)

# Company Selection
company = st.selectbox(
    "Select Company",
    sorted(df["Company"].unique())
)

company_df = df[df["Company"] == company]

# Date Selection
selected_date = st.selectbox(
    "Select Trading Date",
    company_df["Date"].dt.strftime("%Y-%m-%d")
)

row = company_df[
    company_df["Date"] == pd.to_datetime(selected_date)
].iloc[0]

st.subheader("Selected Stock Data")

col1, col2 = st.columns(2)

with col1:
    st.write("**Open:**", row["Open"])
    st.write("**High:**", row["High"])
    st.write("**Low:**", row["Low"])
    st.write("**Volume:**", int(row["Volume"]))
    st.write("**Daily Return:**", round(row["Daily_Return"],4))

with col2:
    st.write("**RSI 14:**", round(row["RSI_14"],2))
    st.write("**MACD:**", round(row["MACD"],2))
    st.write("**SMA 5:**", round(row["SMA_5"],2))
    st.write("**SMA 20:**", round(row["SMA_20"],2))

if st.button("Predict Next Day Movement"):

    values = [[
        row["Open"],
        row["High"],
        row["Low"],
        row["Volume"],
        row["Daily_Return"],
        row["RSI_14"],
        row["MACD"],
        row["SMA_5"],
        row["SMA_20"]
    ]]

    values = scaler.transform(values)

    prediction = model.predict(values)[0]

    probability = model.predict_proba(values)[0]

    st.divider()

    if prediction == 1:
        st.success("📈 Prediction: Stock price is likely to Increase")
    else:
        st.error("📉 Prediction: Stock price is likely to Decrease")

    st.write(f"**Probability of Increase:** {probability[1]*100:.2f}%")
    st.write(f"**Probability of Decrease:** {probability[0]*100:.2f}%")