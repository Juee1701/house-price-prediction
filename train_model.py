from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_PATH = DATA_DIR / "house_prices.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"
RESULTS_PATH = MODEL_DIR / "model_metrics.csv"

FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years", "location"]
NUMERIC_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
CATEGORICAL_FEATURES = ["location"]


def make_demo_data(n=900, seed=42):
    """Generate illustrative data only so the app runs before a real dataset is added."""
    rng = np.random.default_rng(seed)
    locations = np.array(["Central", "North", "South", "East", "West"])
    location = rng.choice(locations, size=n, p=[0.18, 0.20, 0.22, 0.20, 0.20])
    area = rng.integers(450, 3500, size=n).astype(float)
    bedrooms = np.clip(np.round(area / 550 + rng.normal(0, 0.5, n)), 1, 6).astype(int)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, size=n), 1, 5).astype(int)
    age = rng.integers(0, 45, size=n)
    location_effect = pd.Series(location).map({
        "Central": 1800000, "North": 500000, "South": 900000, "East": 250000, "West": 700000
    }).to_numpy()
    price = (
        1800000 + area * 2400 + bedrooms * 250000 + bathrooms * 180000
        - age * 28000 + location_effect + rng.normal(0, 650000, n)
    )
    price = np.maximum(price, 800000)
    return pd.DataFrame({
        "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age, "location": location, "price": price
    })


def normalize_columns(raw):
    """Convert common Indian/Ames-style column names into the app's standard schema."""
    df = raw.copy()
    lower_map = {str(c).strip().lower().replace(" ", "").replace("_", ""): c for c in df.columns}

    def find(*candidates):
        for name in candidates:
            key = name.lower().replace(" ", "").replace("_", "")
            if key in lower_map:
                return lower_map[key]
        return None

    price_col = find("price", "saleprice", "selling_price", "house_price", "target", "amount")
    area_col = find("area_sqft", "area", "squarefeet", "sqft", "grlivarea", "livingarea", "total_sqft")
    bed_col = find("bedrooms", "bedroom", "beds", "bhk", "bedroomabvgr")
    bath_col = find("bathrooms", "bathroom", "baths", "fullbath")
    age_col = find("age_years", "houseage", "age", "yearbuilt", "year_built")
    loc_col = find("location", "neighborhood", "locality", "city", "area_name")

    # Common Indian housing datasets may encode size as total_sqft and target as price.
    if price_col is None or area_col is None:
        raise ValueError(
            "Could not identify the required price and area columns. "
            "Rename your CSV columns to price, area_sqft, bedrooms, bathrooms, age_years, location."
        )

    out = pd.DataFrame()
    out["price"] = pd.to_numeric(df[price_col], errors="coerce")
    out["area_sqft"] = pd.to_numeric(df[area_col], errors="coerce")
    out["bedrooms"] = pd.to_numeric(df[bed_col], errors="coerce") if bed_col else np.nan
    out["bathrooms"] = pd.to_numeric(df[bath_col], errors="coerce") if bath_col else np.nan

    if age_col:
        age_values = pd.to_numeric(df[age_col], errors="coerce")
        # If this looks like a build year, convert year built into approximate age.
        if age_col.lower().replace("_", "") in ["yearbuilt"]:
            out["age_years"] = 2026 - age_values
        else:
            out["age_years"] = age_values
    else:
        out["age_years"] = np.nan

    out["location"] = df[loc_col].astype(str) if loc_col else "Unknown"
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["price", "area_sqft"])
    out = out[(out["price"] > 0) & (out["area_sqft"] > 0)]
    out["location"] = out["location"].fillna("Unknown").astype(str)
    return out


def load_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        raw = pd.read_csv(DATA_PATH)
        normalized = normalize_columns(raw)
        if len(normalized) < 30:
            warnings.warn("Dataset has fewer than 30 usable rows; metrics may be unreliable.")
        normalized.attrs["demo_data"] = False
        return normalized
    demo = make_demo_data()
    demo.attrs["demo_data"] = True
    return demo


def prepare_training_data(df):
    cleaned = df.copy()
    for col in NUMERIC_FEATURES:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned["location"] = cleaned["location"].fillna("Unknown").astype(str)
    cleaned["price"] = pd.to_numeric(cleaned["price"], errors="coerce")
    cleaned = cleaned.dropna(subset=["price", "area_sqft"])
    cleaned = cleaned[(cleaned["price"] > 0) & (cleaned["area_sqft"] > 0)]
    return cleaned


def train_and_save_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = prepare_training_data(load_dataset())
    if len(df) < 20:
        raise ValueError("At least 20 usable dataset rows are recommended to train and test models.")

    X = df[FEATURES]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Preprocessing is fitted only on training data to reduce data leakage.
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    model_definitions = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=180, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    fitted_models = {}
    results = []
    for name, estimator in model_definitions.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])
        # Each pipeline fits its own preprocessing object during fit.
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        results.append({
            "Model": name,
            "MAE": float(mean_absolute_error(y_test, predictions)),
            "RMSE": rmse,
            "R2": float(r2_score(y_test, predictions)),
        })
        fitted_models[name] = pipe

    results_df = pd.DataFrame(results).sort_values("MAE")
    results_df.to_csv(RESULTS_PATH, index=False)

    data_info = {
        "rows": int(len(df)),
        "demo_data": bool(df.attrs.get("demo_data", False)),
    }
    if DATA_PATH.exists():
        data_info["demo_data"] = False

    locations = sorted(df["location"].astype(str).unique().tolist())
    if not locations:
        locations = ["Unknown"]

    bundle = {
        "models": fitted_models,
        "feature_names": FEATURES,
        "location_values": locations,
        "currency": "INR",
        "data_info": data_info,
    }
    joblib.dump(bundle, MODEL_PATH)
    return results_df


if __name__ == "__main__":
    print("Loading dataset and training models...")
    result = train_and_save_models()
    print("\nModel evaluation:")
    print(result.to_string(index=False))
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved metrics to: {RESULTS_PATH}")
    if not DATA_PATH.exists():
        print("\nNOTE: No data/house_prices.csv was found. Demo data was used.")
        print("Add a real dataset and run this script again before reporting project results.")
