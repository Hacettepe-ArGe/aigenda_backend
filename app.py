
from flask import Flask, request, jsonify
from predict import predict_priority_and_time

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    user_id = data.get("user_id", "default")
    result = predict_priority_and_time(text, user_id)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
