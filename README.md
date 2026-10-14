# Smart House Price Prediction and Analysis System

A college-level Machine Learning project built with Python, Scikit-learn, Streamlit, Pandas, Matplotlib and Seaborn.

## Features
- House price prediction website
- Linear Regression, Decision Tree and Random Forest
- MAE, RMSE and R² model evaluation
- Price distribution chart
- Area-versus-price scatter plot
- Average price by location
- Numeric feature correlation heatmap
- Demo-data fallback so the website can be tested before adding a real dataset

## 1. Install Python
Install Python 3.10 or newer from https://www.python.org/downloads/
During installation on Windows, select **Add Python to PATH** if the option is shown.

## 2. Open the project folder
Extract the ZIP file. Open the `House_Price_Prediction_Project` folder in VS Code, or open Command Prompt in that folder.

## 3. Create a virtual environment (recommended)
```powershell
python -m venv .venv
.venv\Scripts\activate
```

If PowerShell blocks activation, use Command Prompt instead:
```bat
.venv\Scripts\activate.bat
```
You can also skip the virtual environment and use `python -m pip` in the next step.

## 4. Install packages
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 5. Run model training
```powershell
python train_model.py
```
If no dataset is present, the script creates an illustrative demo dataset in memory so you can test the project. The generated model and metrics are saved in the `models` folder.

## 6. Launch the website
```powershell
python -m streamlit run app.py
```
Streamlit should open a browser tab, usually at http://localhost:8501.

## 7. Add a real dataset
Place your CSV at:
`data/house_prices.csv`

The program attempts to recognize common column names. For the most reliable result, use exactly these column names:
- `price` — target selling price
- `area_sqft` — living area in square feet
- `bedrooms` — number of bedrooms
- `bathrooms` — number of bathrooms
- `age_years` — age of property in years
- `location` — neighborhood or locality

Example CSV header:
```csv
price,area_sqft,bedrooms,bathrooms,age_years,location
4500000,1200,2,2,10,North
6500000,1500,3,2,5,Central
```
These example records are illustrative only. Do not use only two rows as a real training dataset.

If your data has `SalePrice`, `GrLivArea`, `BedroomAbvGr`, `FullBath`, `YearBuilt`, and `Neighborhood` columns (as in the Kaggle Ames Housing dataset), the script maps them automatically. `YearBuilt` is converted to approximate property age using 2026 as the reference year; update that reference in `train_model.py` if needed.

For the common Bengaluru house-price dataset, inspect its columns carefully. `total_sqft` may contain ranges or strings such as `1200 - 1500`, and `size` may contain text such as `2 BHK`; clean these fields and map them to the standard columns before running this app.

After adding or replacing the CSV:
1. Stop Streamlit with `Ctrl+C`.
2. Run `python train_model.py`.
3. Start the website again with `python -m streamlit run app.py`.

## Important limitations
- Demo data is synthetic and is only for testing the application. Its predictions and scores are not real-market results.
- Price currency is displayed as INR in this starter project. If your dataset uses another currency, update the `currency` setting in `train_model.py` and the UI formatting in `app.py`.
- Predictions are estimates, not professional valuations.
- Keep train/test separation and avoid using test data for preprocessing or model selection.
