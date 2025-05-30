from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')  # Only if you used one

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)  # Uncomment if using scaler
    prediction = model.predict(data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
