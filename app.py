import pickle
from flask import Flask, request
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load all required models and transformers
with open("models/lgbm_pipeline_model.pkl", "rb") as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

print("Loaded selected features:", selected_features)

app = Flask(__name__)

@app.route("/")
def index():
    return {
        "status": "SUCCESS",
        "message": "Service is up"
    }, 200

@app.route('/predict')
def predict():
    try:
        # Get parameters from request
        args = request.args
        
        # Initialize input data with zeros
        input_data = {feature: 0.0 for feature in selected_features}
        
        # Update with provided values
        for feature in selected_features:
            value = args.get(feature)
            if value is not None:
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    return {
                        "status": "ERROR",
                        "message": f"Invalid value for feature {feature}"
                    }, 400

        # Create DataFrame with selected features only
        input_df = pd.DataFrame([input_data])
        
        # Scale the input data
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        return {
            'status': 'SUCCESS',
            'prediction': int(prediction[0])
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'message': str(e)
        }, 400

if __name__ == '__main__':
    app.run(debug=True)