from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Smart House Price Prediction and Analysis System
# Main website file: app.py
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple custom styling
st.markdown(
    """
    <style>
    .stApp { background-color: #f7f7f2; }
    [data-testid="stSidebar"] { background-color: #e9eee8; }
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #dfece5, #f2eee2);
        border: 1px solid #d5dfd7;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: #203b32; margin-bottom: .4rem; }
    .hero p { color: #43564d; font-size: 1.05rem; }
    div[data-testid="stMetric"] {
        background: white; padding: 16px; border-radius: 12px;
        border: 1px solid #e4e7e1;
    }
    .small-note { color: #68736c; font-size: .9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "house_prices.csv"

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location"]
TARGET = "price"
LOCATIONS = ["Katol", "Nagpur", "Pune", "Mumbai", "Nashik", "Other"]


@st.cache_data
def make_demo_data(n=350):
    """Create clearly labelled demo data when no real CSV is supplied."""
    rng = np.random.default_rng(42)
    locations = rng.choice(
        ["Katol", "Nagpur", "Pune", "Mumbai", "Nashik", "Other"],
        size=n,
        p=[0.08, 0.25, 0.22, 0.20, 0.15, 0.10],
    )
    area = rng.integers(450, 3200, size=n)
    bedrooms = np.clip(np.round(area / 500 + rng.normal(0, 0.7, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, size=n), 1, 5).astype(int)
    age = rng.integers(0, 35, size=n)

    location_add = {
        "Katol": 0,
        "Nagpur": 700_000,
        "Pune": 1_600_000,
        "Mumbai": 3_200_000,
        "Nashik": 900_000,
        "Other": 400_000,
    }
    base = (
        area * 2600
        + bedrooms * 180_000
        + bathrooms * 100_000
        - age * 22_000
        + np.array([location_add[x] for x in locations])
        + rng.normal(0, 350_000, n)
    )
    prices = np.maximum(base, 500_000).round(-3).astype(int)
    return pd.DataFrame(
        {
            "price": prices,
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "location": locations,
        }
    )


def load_data():
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            # Allow a few common alternative column names
            aliases = {
                "price": ["price", "house_price", "sale_price", "Price"],
                "area_sqft": ["area_sqft", "area", "square_feet", "sqft", "Area"],
                "bedrooms": ["bedrooms", "bhk", "beds", "Bedroom"],
                "bathrooms": ["bathrooms", "baths", "Bathroom"],
                "age_years": ["age_years", "age", "house_age", "Age"],
                "location": ["location", "city", "area_name", "Location"],
            }
            rename = {}
            for standard, candidates in aliases.items():
                for candidate in candidates:
                    if candidate in df.columns:
                        rename[candidate] = standard
                        break
            df = df.rename(columns=rename)
            if all(col in df.columns for col in FEATURES + [TARGET]):
                df = df[FEATURES + [TARGET]].copy()
                df["location"] = df["location"].fillna("Other").astype(str)
                for col in ["price", "area_sqft", "bedrooms", "bathrooms", "age_years"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=[TARGET, "area_sqft"])
                if len(df) >= 10:
                    return df, False
        except Exception:
            pass
    return make_demo_data(), True


def build_pipeline(model_name):
    numeric_features = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
    categorical_features = ["location"]
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_features),
            ("categorical", categorical_pipe, categorical_features),
        ]
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=120, max_depth=12, random_state=42
        ),
    }
    return Pipeline([("preprocess", preprocess), ("model", models[model_name])])


@st.cache_resource
def train_models(data_fingerprint, df):
    # data_fingerprint makes the cache change when the dataset changes
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    if len(df) < 15:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    fitted = {}
    metric_rows = []
    for name in ["Linear Regression", "Decision Tree", "Random Forest"]:
        model = build_pipeline(name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        fitted[name] = model
        metric_rows.append(
            {
                "Model": name,
                "MAE (₹)": mean_absolute_error(y_test, predictions),
                "RMSE (₹)": np.sqrt(mean_squared_error(y_test, predictions)),
                "R² Score": r2_score(y_test, predictions) if len(y_test) > 1 else 0,
            }
        )
    return fitted, pd.DataFrame(metric_rows)


df, is_demo = load_data()
fingerprint = str(pd.util.hash_pandas_object(df, index=True).sum())
models, metrics = train_models(fingerprint, df)

st.sidebar.title("🏡 House Price Predictor")
st.sidebar.caption("Property Price Analysis")
page = st.sidebar.radio(
    "Navigate",
    ["Home Dashboard", "Predict Price", "Data Visualization", "Model Comparison"],
)
st.sidebar.markdown("---")
st.sidebar.caption("House Price Prediction and Analysis System")

if is_demo:
    st.sidebar.warning(
        "Demo data is being used. Predictions are for demonstration only, not verified market prices."
    )
else:
    st.sidebar.success("Your dataset is loaded.")


def money(value):
    return f"₹{value:,.0f}"


if page == "Home Dashboard":
    st.markdown(
        """
        <div class="hero">
          <h1>Smart House Price Prediction</h1>
          <p>Explore housing data, estimate a property's price, and compare machine-learning models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "This educational mini-project demonstrates how property features can be used by a machine-learning model."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties in dataset", f"{len(df):,}")
    c2.metric("Average price", money(df[TARGET].mean()))
    c3.metric("Average area", f"{df['area_sqft'].mean():,.0f} sq ft")
    c4.metric("Locations", f"{df['location'].nunique()}")

    left, right = st.columns(2)
    with left:
        st.subheader("Price distribution")
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[TARGET], bins=25, kde=True, ax=ax)
        ax.set_xlabel("House price (₹)")
        ax.set_ylabel("Number of properties")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Average price by location")
        avg = df.groupby("location", as_index=False)[TARGET].mean().sort_values(TARGET)
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.barplot(data=avg, x=TARGET, y="location", ax=ax)
        ax.set_xlabel("Average price (₹)")
        ax.set_ylabel("Location")
        st.pyplot(fig)
        plt.close(fig)

elif page == "Predict Price":
    st.markdown(
        """
        <div class="hero">
          <h1>Predict a House Price</h1>
          <p>Enter the property details below and choose a model for an estimated price.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    locations_in_data = sorted(df["location"].dropna().astype(str).unique().tolist())
    location_options = sorted(set(LOCATIONS + locations_in_data))
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            area = st.number_input("Area (square feet)", min_value=100, max_value=20000, value=1200, step=50)
            bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)
            bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4, 5], index=1)
        with right:
            age = st.number_input("Property age (years)", min_value=0, max_value=150, value=5, step=1)
            location = st.selectbox("Location", location_options, index=location_options.index("Nagpur") if "Nagpur" in location_options else 0)
            model_name = st.selectbox("Machine-learning model", list(models.keys()), index=2)
        submitted = st.form_submit_button("Predict House Price", use_container_width=True)

    if submitted:
        input_df = pd.DataFrame(
            [{
                "area_sqft": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age_years": age,
                "location": location,
            }]
        )
        prediction = max(0, float(models[model_name].predict(input_df)[0]))
        st.success("Prediction completed")
        st.metric("Estimated house price", money(prediction))
        st.caption(f"Model used: {model_name}")
        st.warning(
            "This is a model-generated estimate. Actual prices depend on property condition, exact locality, market conditions, and reliable training data."
        )
        budget = st.number_input("Optional: enter your budget (₹)", min_value=0, value=0, step=100000)
        if budget > 0:
            if prediction <= budget:
                st.success(f"The estimate is within your budget by {money(budget - prediction)}.")
            else:
                st.info(f"The estimate exceeds your budget by {money(prediction - budget)}.")

elif page == "Data Visualization":
    st.markdown(
        """
        <div class="hero">
          <h1>Data Visualization</h1>
          <p>Understand how area, location, and other property features relate to price.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chart_choice = st.selectbox(
        "Choose a chart",
        [
            "Area vs Price",
            "Price Distribution",
            "Average Price by Location",
            "Correlation Heatmap",
        ],
    )
    if chart_choice == "Area vs Price":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x="area_sqft", y=TARGET, hue="location", alpha=0.75, ax=ax)
        ax.set_xlabel("Area (square feet)")
        ax.set_ylabel("House price (₹)")
        ax.set_title("Area vs House Price")
        st.pyplot(fig)
        plt.close(fig)
    elif chart_choice == "Price Distribution":
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df[TARGET], bins=30, kde=True, ax=ax)
        ax.set_xlabel("House price (₹)")
        ax.set_title("Distribution of House Prices")
        st.pyplot(fig)
        plt.close(fig)
    elif chart_choice == "Average Price by Location":
        avg = df.groupby("location", as_index=False)[TARGET].mean().sort_values(TARGET, ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=avg, x=TARGET, y="location", ax=ax)
        ax.set_xlabel("Average house price (₹)")
        ax.set_title("Average House Price by Location")
        st.pyplot(fig)
        plt.close(fig)
    else:
        numeric = df[["price", "area_sqft", "bedrooms", "bathrooms", "age_years"]].corr()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric, annot=True, cmap="Greens", fmt=".2f", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("View dataset"):
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download current dataset as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="house_prices_dataset.csv",
            mime="text/csv",
        )

elif page == "Model Comparison":
    st.markdown(
        """
        <div class="hero">
          <h1>Model Comparison</h1>
          <p>Compare three regression algorithms using standard evaluation metrics.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("MAE and RMSE should generally be lower; R² is closer to 1 when the model fits better.")
    display_metrics = metrics.copy()
    display_metrics["MAE (₹)"] = display_metrics["MAE (₹)"].map(lambda x: f"₹{x:,.0f}")
    display_metrics["RMSE (₹)"] = display_metrics["RMSE (₹)"].map(lambda x: f"₹{x:,.0f}")
    display_metrics["R² Score"] = display_metrics["R² Score"].map(lambda x: f"{x:.3f}")
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    left, right = st.columns(2)
    with left:
        st.subheader("RMSE comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=metrics, x="Model", y="RMSE (₹)", ax=ax)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("RMSE (₹)")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("R² score comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=metrics, x="Model", y="R² Score", ax=ax)
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("R² score")
        st.pyplot(fig)
        plt.close(fig)

    best = metrics.sort_values("RMSE (₹)").iloc[0]
    st.info(f"On the current dataset, **{best['Model']}** has the lowest RMSE.")
    st.caption(
        "Model scores depend on the dataset and the train/test split. Demo-data scores are for educational demonstration only."
    )

st.markdown("---")
st.markdown(
    "<p class='small-note'>House Price Prediction and Analysis System</p>",
    unsafe_allow_html=True,
)
