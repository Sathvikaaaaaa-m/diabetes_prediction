from flask import Flask, request, jsonify
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load your model and scaler once when app starts
with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Assuming input features come as a list in JSON like {"features": [val1, val2, ...]}
        features = data['features']
        features_np = np.array(features).reshape(1, -1)

        # Scale the features
        scaled_features = scaler.transform(features_np)

        # Get prediction from the model
        prediction = model.predict(scaled_features)

        # Return prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400
import os
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

