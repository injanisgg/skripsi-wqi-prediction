import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load all required models and transformers
with open("app/models/lgbm_pipeline_model.pkl", "rb") as f:
    model = pickle.load(f)

with open('app/models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('app/models/selected_features.pkl', 'rb') as f:
    selected_features = pickle.load(f)

print("Loaded selected features:", selected_features)

# Tentukan folder template
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

@app.route("/")
def index():
    return render_template('index.html')

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