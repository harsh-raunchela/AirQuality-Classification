from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# 1. Load trained model
model = joblib.load('knn.pkl')

# Automatically extract feature names from the trained model
if hasattr(model, 'feature_names_in_'):
    REQUIRED_FEATURES = list(model.feature_names_in_)
    print("\n✅ Model expects these exact feature names:", REQUIRED_FEATURES, "\n")
else:
    # default feature list
    REQUIRED_FEATURES = [
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
    """Safely convert inputs to float, defaulting empty or invalid values to 0.0."""
    try:
        return float(val) if val not in (None, "") else 0.0
    except (ValueError, TypeError):
        return 0.0

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        
        normalized_inputs = {str(k).strip().lower(): v for k, v in data.items()}
        
        # Construct vector matching the EXACT model column names and order
        input_vector = []
        for feature_name in REQUIRED_FEATURES:
            # Match by exact feature name first, then by lowercase
            val = data.get(feature_name)
            if val is None:
                val = normalized_inputs.get(feature_name.strip().lower(), 0.0)
            
            input_vector.append(safe_float(val))

        # DataFrame using exact feature names expected by the model
        input_df = pd.DataFrame([input_vector], columns=REQUIRED_FEATURES)

        # Run inference
        prediction = model.predict(input_df)[0]
        
        category = CATEGORY_MAP.get(prediction, str(prediction))
        return jsonify({'category': category})

    except Exception as e:
        print("Backend Error Trace:", str(e))
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)