import os
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from train_model import (
    DATA_PATH, MODEL_PATH, RESULTS_PATH, load_dataset,
    prepare_training_data, train_and_save_models
)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Smart House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple custom styling
st.markdown("""
<style>
.main .block-container {padding-top: 1.8rem; padding-bottom: 2.5rem;}
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 16px;
    background: linear-gradient(120deg, #edf4ff, #f6f0ff);
    color: #17233b;
    margin-bottom: 1rem;
}
.small-note {font-size: 0.9rem; color: #666;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        train_and_save_models()
    return joblib.load(MODEL_PATH)

@st.cache_data
def get_data():
    return load_dataset()

def format_price(value, currency):
    if currency == "INR":
        return f"₹{value:,.0f}"
    return f"${value:,.0f}"

bundle = load_model_bundle()
data_info = bundle["data_info"]
feature_names = bundle["feature_names"]
location_values = bundle["location_values"]
target_currency = bundle.get("currency", "INR")
models = bundle["models"]
metrics = pd.read_csv(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()

with st.sidebar:
    st.title("🏡 House Price Predictor")
    st.caption("Property Price Analysis")
    page = st.radio(
        "Navigate",
        ["Home Dashboard", "Predict Price", "Data Visualization", "Model Comparison"],
    )
    st.divider()
    st.caption("House Price Prediction and Analysis System")
    if data_info.get("demo_data", False):
        st.warning("Demo data is active. Predictions are illustrative, not real market estimates.")
    else:
        st.success("Using the dataset found in the data folder.")

if page == "Home Dashboard":
    st.markdown("""
    <div class="hero">
      <h1>Smart House Price Prediction</h1>
      <p>Estimate property prices with Machine Learning and explore the factors
      associated with house values.</p>
    </div>
    """, unsafe_allow_html=True)

    if data_info.get("demo_data", False):
        st.warning(
            "The website is currently using generated demo data so you can test every page. "
            "Replace it with a real dataset before using the results in your college report."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset records", f"{data_info.get('rows', 0):,}")
    c2.metric("Input features", str(len(feature_names)))
    c3.metric("ML algorithms", str(len(models)))
    c4.metric("Currency", target_currency)

    st.subheader("Project objectives")
    left, right = st.columns(2)
    with left:
        st.markdown("""
        - Predict an estimated house price from property details.
        - Compare Linear Regression, Decision Tree and Random Forest.
        - Explore relationships between property features and price.
        """)
    with right:
        st.markdown("""
        - Evaluate models using MAE, RMSE and R².
        - Present results in a simple interactive website.
        - Demonstrate a complete data-to-prediction ML workflow.
        """)
    st.subheader("How it works")
    st.write("Dataset → Cleaning and preprocessing → Model training → Evaluation → Website prediction")
    st.info("Use the sidebar to open the prediction form, graphs, and model comparison.")

elif page == "Predict Price":
    st.title("🏠 Predict a House Price")
    st.write("Enter the property details below and select **Predict price**.")
    if data_info.get("demo_data", False):
        st.info("Demo mode: the model was trained on generated sample data, not actual property listings.")

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Living area (square feet)", min_value=100.0, max_value=50000.0,
                               value=1500.0, step=50.0)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)
    with col2:
        age = st.number_input("House age (years)", min_value=0, max_value=200, value=10, step=1)
        default_location = location_values[0] if location_values else "Unknown"
        location = st.selectbox("Location / neighborhood", location_values, index=0)
        model_choice = st.selectbox("Prediction model", list(models.keys()), index=0)

    if st.button("Predict price", type="primary", use_container_width=True):
        row = pd.DataFrame([{
            "area_sqft": area,
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "age_years": int(age),
            "location": location,
        }])
        try:
            prediction = float(models[model_choice].predict(row)[0])
            st.success(f"Estimated house price: {format_price(max(0, prediction), target_currency)}")
            st.caption("This is a model estimate, not a professional valuation or guaranteed market price.")
            budget = st.number_input("Optional: enter your budget", min_value=0.0, value=0.0, step=100000.0,
                                     key="budget_input")
            if budget > 0:
                difference = budget - prediction
                if difference >= 0:
                    st.write(f"Your budget is approximately {format_price(difference, target_currency)} above the estimate.")
                else:
                    st.write(f"You may need approximately {format_price(abs(difference), target_currency)} more than your budget.")
        except Exception as exc:
            st.error(f"Could not make a prediction. Check the dataset and model features. Details: {exc}")

elif page == "Data Visualization":
    st.title("📊 Data Visualization")
    df = get_data()
    st.write("Explore the dataset used by the project.")
    if data_info.get("demo_data", False):
        st.warning("All charts currently show generated demo data.")
    df = df.copy()
    if "price" not in df.columns:
        st.error("The dataset does not contain a recognized price column.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("House price distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df["price"].dropna(), bins=30, kde=True, ax=ax)
        ax.set_xlabel(f"Price ({target_currency})")
        ax.set_ylabel("Number of houses")
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.subheader("Area vs house price")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.scatterplot(data=df, x="area_sqft", y="price", hue="location", legend=False, ax=ax, alpha=0.65)
        ax.set_xlabel("Living area (square feet)")
        ax.set_ylabel(f"Price ({target_currency})")
        st.pyplot(fig)
        plt.close(fig)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Average price by location")
        location_summary = df.groupby("location", dropna=False)["price"].mean().sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        location_summary.sort_values().plot(kind="barh", ax=ax)
        ax.set_xlabel(f"Average price ({target_currency})")
        ax.set_ylabel("Location")
        st.pyplot(fig)
        plt.close(fig)
    with c4:
        st.subheader("Feature correlation heatmap")
        numeric_cols = [c for c in ["area_sqft", "bedrooms", "bathrooms", "age_years", "price"] if c in df.columns]
        corr = df[numeric_cols].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("View dataset sample"):
        st.dataframe(df.head(100), use_container_width=True)

elif page == "Model Comparison":
    st.title("🤖 Machine Learning Model Comparison")
    if data_info.get("demo_data", False):
        st.warning("Metrics are based on generated demo data. Replace the data before reporting scores.")
    if metrics.empty:
        st.info("No evaluation results found. Run train_model.py to generate them.")
    else:
        st.subheader("Evaluation metrics")
        st.dataframe(metrics.round(3), use_container_width=True)
        st.caption("Lower MAE/RMSE is better. Higher R² is generally better.")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("MAE by model")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=metrics, x="Model", y="MAE", ax=ax)
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.subheader("R² score by model")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(data=metrics, x="Model", y="R2", ax=ax)
            ax.tick_params(axis="x", rotation=20)
            st.pyplot(fig)
            plt.close(fig)
        best = metrics.sort_values("MAE").iloc[0]
        st.info(f"Best model by MAE on the test set: **{best['Model']}** (MAE = {best['MAE']:,.2f}).")
    if st.button("Retrain models using current dataset"):
        with st.spinner("Training models..."):
            train_and_save_models()
            st.cache_resource.clear()
            st.cache_data.clear()
        st.success("Models retrained. Refresh the page if the updated model is not shown immediately.")
        st.rerun()
