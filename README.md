---

# Predictive Modeling for Voltage, Amperage, and Power

This Python script (`jagodata-python-reg.py`) demonstrates a predictive modeling workflow for forecasting voltage (`VOLT`), amperage (`AMPERE`), and power (`POWER`) based on historical data using machine learning techniques.

## Libraries Used

The script utilizes the following libraries:
- `pandas` for data manipulation and analysis.
- `matplotlib.pyplot` for data visualization.
- `scikit-learn` (`sklearn`) for model selection (`train_test_split`), hyperparameter tuning (`GridSearchCV`), model evaluation (`RandomForestRegressor`, `mean_absolute_error`, `mean_squared_error`, `mean_absolute_percentage_error`), and preprocessing (`StandardScaler`).
- `numpy` for numerical operations.
- `time` for tracking execution time.
- `warnings` for managing warnings during execution.

## Steps Covered in the Script

1. **Data Loading and Initial Exploration**
   - Reads data from a CSV file (`DATA TA FIXS MARET APRIL.csv`).
   - Renames columns and formats the `DATE_TIME` column to datetime format.

2. **Data Preprocessing**
   - Adds lagged features (`IRRADIANCE_LAG_n`) for `IRRADIANCE` using a specified lag window.
   - Drops rows with missing values (`NaN`) after adding lagged features.

3. **Data Splitting and Standardization**
   - Splits the data into training and testing sets.
   - Standardizes numerical features using `StandardScaler`.

4. **Model Training and Evaluation**
   - Trains `RandomForestRegressor` models for each target (`VOLT`, `AMPERE`, `POWER`).
   - Evaluates models using Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

5. **Hyperparameter Tuning**
   - Uses `GridSearchCV` to find optimal hyperparameters for `RandomForestRegressor` models.
   - Re-trains models with the best hyperparameters found during the tuning process.

6. **Prediction on New Data**
   - Reads and preprocesses new data (`DATA TA FIXS APRIL MEI.csv`).
   - Uses trained models to predict `VOLT`, `AMPERE`, and `POWER` for future periods.
   - Evaluates predictions using MSE, MAE, and RMSE.

7. **Output**
   - Generates predictions and actual values for a specified date range (`2024-04-12` to `2024-05-11`).
   - Outputs metrics and results for comparison and analysis.

## Usage
To run the script:
```bash
python jagodata-python-reg.py
```
Ensure the required CSV files (`DATA TA FIXS MARET APRIL.csv`, `DATA TA FIXS APRIL MEI.csv`) are in the same directory as the script.

---