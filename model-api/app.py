from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load your model and scaler once when app starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
