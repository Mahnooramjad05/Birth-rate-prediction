
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 Machine Learning & Time-Series Forecasting Dashboard")

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Data Description")
    st.write(data.describe())

    columns = data.columns.tolist()
    numeric_cols = data.select_dtypes(include='number').columns.tolist()

    # === FEATURE SELECTION ===
    st.header("🧠 Feature Selection")
    target_col = st.selectbox("Select target column", options=numeric_cols)
    features = st.multiselect("Select features", options=[col for col in numeric_cols if col != target_col])

    # === PREPROCESSING ===
    st.header("🧹 Data Preprocessing")
    scale_method = st.selectbox("Select a scaling method", options=["None", "Standard", "MinMax", "Robust", "MaxAbs", "Normalize"])
    if scale_method != "None":
        scaler = {
            "Standard": StandardScaler(),
            "MinMax": MinMaxScaler(),
            "Robust": RobustScaler(),
            "MaxAbs": MaxAbsScaler(),
            "Normalize": Normalizer()
        }[scale_method]

        data[features] = scaler.fit_transform(data[features])
        st.success(f"✅ Applied {scale_method} scaling")

    # === VISUALIZATION ===
    st.header("📈 Visualization")
    plot_type = st.selectbox("Choose plot type", options=["Scatter", "Line", "Box", "Histogram", "Heatmap"])
    col1 = st.selectbox("X-axis", options=features)
    col2 = st.selectbox("Y-axis", options=[target_col] + features)

    fig, ax = plt.subplots()
    try:
        if plot_type == "Scatter":
            sns.scatterplot(x=data[col1], y=data[col2], ax=ax)
        elif plot_type == "Line":
            sns.lineplot(x=data[col1], y=data[col2], ax=ax)
        elif plot_type == "Box":
            sns.boxplot(y=data[col2], ax=ax)
        elif plot_type == "Histogram":
            sns.histplot(data[col2], bins=30, kde=True, ax=ax)
        elif plot_type == "Heatmap":
            corr = data[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Plotting error: {e}")

    # === MODELING ===
    st.header("🔮 Time Series Forecasting")
    model_choice = st.selectbox("Select Model", ["XGBoost", "LSTM", "GRU", "ARIMA", "Prophet"])
    lookback = st.slider("Lookback window", 1, 100, 10)
    future_steps = st.slider("Steps to Forecast", 1, 50, 10)

    def prepare_series(series, lookback):
        X, y = [], []
        for i in range(len(series) - lookback):
            X.append(series[i:i + lookback])
            y.append(series[i + lookback])
        return np.array(X), np.array(y)

    if st.button("Train Model"):
        y = data[target_col].values
        X, y_seq = prepare_series(y, lookback)
        trained_model = None

        if model_choice == "XGBoost":
            X_train, X_test, y_train, y_test = train_test_split(X, y_seq, test_size=0.2)
            model = xgb.XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.success(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, R²: {r2_score(y_test, y_pred):.4f}")
            trained_model = model

        elif model_choice == "LSTM":
            X = X.reshape((X.shape[0], X.shape[1], 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y_seq, test_size=0.2)
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(lookback, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mae')
            model.fit(X_train, y_train, epochs=10, verbose=0)
            y_pred = model.predict(X_test).flatten()
            st.success(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, R²: {r2_score(y_test, y_pred):.4f}")
            trained_model = model

        elif model_choice == "GRU":
            X = X.reshape((X.shape[0], X.shape[1], 1))
            X_train, X_test, y_train, y_test = train_test_split(X, y_seq, test_size=0.2)
            model = Sequential([
                GRU(50, activation='relu', input_shape=(lookback, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mae')
            model.fit(X_train, y_train, epochs=10, verbose=0)
            y_pred = model.predict(X_test).flatten()
            st.success(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, R²: {r2_score(y_test, y_pred):.4f}")
            trained_model = model

        elif model_choice == "ARIMA":
            model = ARIMA(y, order=(5, 1, 0))
            fitted = model.fit()
            y_pred = fitted.predict(start=0, end=len(y)-1)
            st.success(f"MAE: {mean_absolute_error(y, y_pred):.4f}, R²: {r2_score(y, y_pred):.4f}")
            trained_model = fitted

        elif model_choice == "Prophet":
            df = pd.DataFrame({
                "ds": pd.date_range(start='2020-01-01', periods=len(y), freq='D'),
                "y": y
            })
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=future_steps)
            forecast = model.predict(future)
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_steps))
            st.line_chart(forecast.set_index("ds")[["yhat"]])
