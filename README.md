# ⚡ WattWise AI: Predictive Grid Dashboard

An AI-driven predictive digital twin for electrical grids. This decoupled machine learning microservice uses a Random Forest Regressor to forecast energy demand 48 hours in advance, allowing grid operators to proactively balance load, minimize Peaker Plant usage, and prevent blackouts.
> **Project Origin:** This architecture was originally prototyped by Team Hackorbit during the **Fusion Techathon 4.0** (Alva's Institute of Engineering and Technology). It was built to solve the "Electricity Demand Forecasting for Smart Grids" problem statement under a strict time constraint of 24hrs.

## 🚀 System Architecture
* **Backend Intelligence:** Scikit-Learn (Random Forest, 50 Estimators)
* **Feature Engineering:** Cyclical time encoding (Sin/Cos), synthetic thermal distributions, and boolean holiday flagging.
* **API Microservice:** Python Flask REST API (CORS enabled)
* **Frontend UI:** Vanilla JS, CSS Splash Animations, and Chart.js for real-time visualization.

---

## 🛠️ How to Run Locally

Follow these steps to get the full predictive pipeline running on your local machine.

### Step 1: Prerequisites
Ensure you have **Python 3.13+** installed. Install the required data science and server libraries using pip:

pip install pandas scikit-learn flask flask-cors holidays numpy

Step 2: Prepare the Data
Place the historical grid telemetry file (AEP_hourly.csv) directly in the root directory of the project.

Step 3: Train the AI Model
Run the model script to perform feature engineering, train the Random Forest, and export the 48-hour predictions to a JSON payload.

python model.py

(Wait for the console to display the MAPE/RMSE scores and the ✅ Done! Data safely exported message).

Step 4: Start the API Server
Launch the Flask backend to serve the predictions to the frontend dashboard.

python app.py
The REST API will now be live and listening at: http://127.0.0.1:5000/api/forecast

Step 5: View the Dashboard
With the Flask server running in the terminal, simply double-click your index.html file to open it in any web browser. The JavaScript will instantly fetch the payload from the local API and render the UI.

Architectural Note: This system utilizes a decoupled microservice pattern. The model (model.py) pre-calculates the AI inference payload, which the lightweight Flask server (app.py) serves dynamically. This ensures the frontend dashboard loads instantly without waiting for model retraining.
