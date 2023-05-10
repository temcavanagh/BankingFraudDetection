from __future__ import unicode_literals
import json
from flask import Flask, request
import joblib

app = Flask("Banking Fraud Detection")

app.model = joblib.load('models/model.pickle')

@app.route('/')
def hello_world():
    print("Banking Fraud Detection")
    return "<p>Banking Fraud Detection</p>"

@app.route("/predict", methods=["POST"])
def predict_fraud():
    input_data = request.get_json()
    if u"features" not in input_data:
        return json.dumps({u"error": u"No input features"}), 400
    if not input_data[u"features"] or not isinstance(input_data[u"features"], list):
        return json.dumps({u"error": u"No feature values available"}), 400
    if isinstance(input_data[u"features"][0], list):
        results = app.model.predict_proba(input_data[u"features"]).tolist()
    else:
        results = app.model.predict_proba([input_data[u"features"]]).tolist()
    return json.dumps({u"scores": [result[1] for result in results]}), 200
