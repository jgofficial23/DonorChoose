from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model + columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

@app.route('/')
def home():
    return "ML Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert input to dataframe
        input_df = pd.DataFrame([data])

        # Align with training columns
        input_df = input_df.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "approval_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)