from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1. Load trained model
model = joblib.load('knn.pkl')

# Feature list matching model training columns
FEATURE_ORDER = [
    'PM2.5', 'NO2', 'NOx', 'NH3', 'SO2', 'CO', 'Ozone', 
    'Benzene', 'Toluene', 'Xylene', 'MP-Xylene', 'AT', 
    'RH', 'WS', 'WD', 'SR', 'BP', 'VWS', 'Month'
]

CATEGORY_MAP = {
    0: "Good",
    1: "Satisfactory",
    2: "Moderate",
    3: "Poor",
    4: "Very Poor",
    5: "Severe"
}

def safe_float(val):
    """Safely convert inputs to float, defaulting empty strings to 0.0."""
    try:
        return float(val) if val not in (None, "") else 0.0
    except (ValueError, TypeError):
        return 0.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        
        # Safely convert incoming inputs
        input_vector = [safe_float(data.get(feature)) for feature in FEATURE_ORDER]
        
        # Create DataFrame with exact column names (fixes UserWarning)
        input_df = pd.DataFrame([input_vector], columns=FEATURE_ORDER)
        
        # Run inference
        prediction = model.predict(input_df)[0]
        
        category = CATEGORY_MAP.get(prediction, str(prediction))
        return jsonify({'category': category})

    except Exception as e:
        print("Backend Error Trace:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)