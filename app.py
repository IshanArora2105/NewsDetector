from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

MODELS_DIR = 'models'

try:
    vectorization = joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    LR = joblib.load(os.path.join(MODELS_DIR, 'logistic_regression_model.pkl'))
    DTC = joblib.load(os.path.join(MODELS_DIR, 'decision_tree_model.pkl'))
    RFC = joblib.load(os.path.join(MODELS_DIR, 'random_forest_model.pkl'))
    GBC = joblib.load(os.path.join(MODELS_DIR, 'gradient_boosting_model.pkl'))
    print("All models and vectorizer loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading models: {e}")
    exit()

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', '', text)
    return text

def output_label(n):
    if n == 0:
        return "FAKE NEWS"
    elif n == 1:
        return "GENUINE NEWS"

@app.route('/')
def home():
    return "Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        news_article = data['news_article']

        cleaned_news = wordopt(news_article)
        new_xv_test = vectorization.transform([cleaned_news])

        pred_lr = LR.predict(new_xv_test)[0]
        pred_dtc = DTC.predict(new_xv_test)[0]
        pred_rfc = RFC.predict(new_xv_test)[0]
        pred_gbc = GBC.predict(new_xv_test)[0]

        label_lr = output_label(pred_lr)
        label_dtc = output_label(pred_dtc)
        label_rfc = output_label(pred_rfc)
        label_gbc = output_label(pred_gbc)

        return jsonify({
            "LR_Prediction": label_lr,
            "DTC_Prediction": label_dtc,
            "RFC_Prediction": label_rfc,
            "GBC_Prediction": label_gbc
        })
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
