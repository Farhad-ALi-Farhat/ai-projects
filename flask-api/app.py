from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Train model
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return "ML API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    value = data['value']

    prediction = model.predict([[value]])
    return jsonify({'prediction': float(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
