import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import holidays
import json

print("Loading Indian Grid Data (Excel takes a moment)...")
# 1. Load the Excel file
df = pd.read_excel('hourlyLoadDataIndia.xlsx')

# 2. BULLETPROOF COLUMN EXTRACTION
# Strip invisible spaces from column names (Classic Excel issue)
df.columns = df.columns.str.strip()

# Dynamically find the columns without relying on exact spelling
date_col = [col for col in df.columns if 'date' in col.lower()][0]
south_col = [col for col in df.columns if 'outhern' in col.lower()][0]

print(f"Success: Found columns '{date_col}' and '{south_col}'")

# Extract only what we need and rename to trick the API
df = df[[date_col, south_col]].copy()
df.rename(columns={date_col: 'Datetime', south_col: 'Megawatts'}, inplace=True)

# 3. Convert to proper time format
df['Datetime'] = pd.to_datetime(df['Datetime'])

print("Extracting Time & Indian Holiday Features...")
# 4. Feature Engineering
df['Hour'] = df['Datetime'].dt.hour
df['DayOfWeek'] = df['Datetime'].dt.dayofweek
df['Month'] = df['Datetime'].dt.month

# THE FLEX: Automatically detects Indian Holidays!
in_holidays = holidays.IN()
df['Is_Holiday'] = df['Datetime'].apply(lambda x: 1 if x in in_holidays else 0)

# Clean up any missing rows
df = df.dropna()

# 5. Set up the Brain
X = df[['Hour', 'DayOfWeek', 'Month', 'Is_Holiday']]
y = df['Megawatts']

# 6. Train the Model
print("Training the V2 Southern Grid AI...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 7. Test Accuracy
predictions = model.predict(X_test)
mape = mean_absolute_percentage_error(y_test, predictions) * 100

print("-" * 50)
print(f"🥇 INDIAN AI RESULTS: {mape:.2f}% Error Rate")
print("-" * 50)

# 8. Export for the Frontend Dashboard
last_48_actual = y_test.iloc[-48:].tolist()
last_48_predicted = predictions[-48:].tolist()

output_data = {
    "mape_score": round(mape, 2),
    "actual_demand": last_48_actual,
    "predicted_demand": last_48_predicted
}

with open('api_data.json', 'w') as f:
    json.dump(output_data, f)
    
print("✅ Done! Indian Grid Data safely exported to api_data.json")