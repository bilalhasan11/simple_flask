from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (you can restrict it later if needed)

# Load the model (adjust path if needed)
MODEL_PATH = "decision_tree_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Model file not found at '{MODEL_PATH}'")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()
        hours_studied = float(data['hours_studied'])
        sleep = float(data['sleep'])
        coffee_cups = float(data['coffee_cups'])

        # Prepare input for the model
        input_data = np.array([[hours_studied, sleep, coffee_cups]])

        # Make prediction
        prediction = model.predict(input_data)[0]
        result = "pass" if prediction == 1 else "fail"

        # Return result as JSON
        return jsonify({"result": result})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5010)  # Match your port