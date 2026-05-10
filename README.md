# techathon-electricity-grid-fusion-techathon
How to Run
Follow these steps to get the full pipeline running on your local machine:
1. Prerequisites
Ensure you have Python 3.13+ installed. Install the required libraries using:
pip install pandas scikit-learn flask flask-cors holidays numpy
2. Prepare the Data
Place the AEP_hourly.csv file in the root directory of the project.
3. Train the AI Model
Run the model script to perform feature engineering, train the Random Forest, and export the predictions to JSON:
python model.py
Wait for the "✅ Done! Data safely exported" message.
4. Start the API Server
Launch the Flask backend to serve the predictions to the frontend:
python app.py
The API will be live at http://127.0.0.1:5000/api/forecast.