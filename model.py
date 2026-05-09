#Electricity Demand Forecasting for Smart Grid 
import pandas as pd #CSV to dataframe
import numpy as np #For trignometric functions 
import holidays
import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'AEP_hourly.csv'))
print("Loading dataset...")


# Load the data
df = pd.read_csv('AEP_hourly.csv')

# Convert to datetime and sort
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)
df = df.drop_duplicates(subset='Datetime').reset_index(drop=True)

print("Enriching data with contextual signals...")

# 1. Time-of-day & Day of week
df['Hour'] = df['Datetime'].dt.hour
df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month

# 2. Public Holidays (Checks if the date is a US holiday, returns 1 for Yes, 0 for No)
us_holidays = holidays.US()
df['Is_Holiday'] = df['Datetime'].dt.date.apply(lambda x: 1 if x in us_holidays else 0)

# 3. Synthetic Temperature Readings (Creates realistic temps based on the season)
def generate_temp(month):
    if month in [12, 1, 2]:   # Winter (Cold)
        return np.random.normal(5, 5) 
    elif month in [3, 4, 5]:  # Spring (Mild)
        return np.random.normal(15, 5)
    elif month in [6, 7, 8]:  # Summer (Hot)
        return np.random.normal(28, 4)
    else:                     # Fall (Mild)
        return np.random.normal(18, 5)
#PRNG
np.random.seed(42)
daily_temps = df.groupby(df['Datetime'].dt.date)['Month'].first().apply(generate_temp)
df['Temperature_C'] = df['Datetime'].dt.date.map(daily_temps).round(1)

# Renaming for clarity
df = df.rename(columns={'AEP_MW': 'Megawatts'})

print("--------------------------------------------------")
print("SUCCESS: Dataset is fully enriched and ready for AI!")
print("--------------------------------------------------")

print(df[['Datetime', 'Megawatts', 'Hour', 'DayOfWeek', 'Is_Holiday', 'Temperature_C']].head(10))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

print("Preparing data for the AI Model...")

# 1. Define the inputs (Features) and the output (Target)
X = df[['Hour', 'DayOfWeek', 'Is_Holiday', 'Temperature_C']]
y = df['Megawatts']

# 2. Split the data into Past (Training) and Future (Testing)
# shuffle=False is CRITICAL for time-series forecasting. We can't use the future to predict the past!
# We are using 80% of the data to train, and holding back the last 20% to test it.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Training the AI Brain (This might take 30-60 seconds)...")

# 3. Create and train the Random Forest model
# We use 50 'trees' to keep the training fast for the hackathon
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("Making predictions for the test period...")

# 4. Predict the future!
predictions = model.predict(X_test)

# 5. Calculate the exact math required by the judges
mape = mean_absolute_percentage_error(y_test, predictions) * 100
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("--------------------------------------------------")
print(f" AI MODEL RESULTS:")
print(f"MAPE Score: {mape:.2f}% (Average error percentage)")
print(f"RMSE Score: {rmse:.2f} Megawatts")
print("--------------------------------------------------")
print("Exporting results for the frontend API...")
# We slice the last 48 hours of data so the frontend has a clean 2-day graph to plot
# .tolist() converts the pandas/numpy arrays into standard Python lists
last_48_actual = y_test.iloc[-48:].tolist()
last_48_predicted = predictions[-48:].tolist()
# Package it exactly how the frontend needs it
output_data = {
    "mape_score": round(mape, 2),
    "rmse_score": round(rmse, 2),
    "actual_demand": last_48_actual,
    "predicted_demand": [round(num, 2) for num in last_48_predicted]
}
# Save it to a file
with open('api_data.json', 'w') as f:
    json.dump(output_data, f)

print("✅ Done! Data safely exported to api_data.json")
import pandas as pd
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nAI Logic - Which features mattered most:")
print(importances.sort_values(ascending=False))