import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib
import os
import subprocess
import sys

required_packages = ['pandas', 'numpy', 'scikit-learn', 'joblib']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing missing package: {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Please install it manually: pip install {package}")
            print(f"Error: {e}")
            sys.exit(1)

true = pd.read_csv('true.csv')
fake = pd.read_csv('fake.csv')

true['label'] = 1
fake['label'] = 0

news = pd.concat([fake, true], axis=0)
news = news.drop(['title', 'subject', 'date'], axis=1)
news = news.sample(frac=1).reset_index(drop=True)

def wordopt(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\n', '', text)
    return text

news['text'] = news['text'].apply(wordopt)

x = news['text']
y = news['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

LR = LogisticRegression(max_iter=1000)
LR.fit(xv_train, y_train)
pred_lr = LR.predict(xv_test)
print("Logistic Regression Score:", LR.score(xv_test, y_test))
print("Logistic Regression Classification Report:")
print(classification_report(y_test, pred_lr))

DTC = DecisionTreeClassifier()
DTC.fit(xv_train, y_train)
pred_dtc = DTC.predict(xv_test)
print("Decision Tree Classifier Score:", DTC.score(xv_test, y_test))
print("Decision Tree Classification Report:")
print(classification_report(y_test, pred_dtc))

rfc = RandomForestClassifier(random_state=42)
rfc.fit(xv_train, y_train)
predict_rfc = rfc.predict(xv_test)
print("Random Forest Classifier Score:", rfc.score(xv_test, y_test))
print("Random Forest Classification Report:")
print(classification_report(y_test, predict_rfc))

gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(xv_train, y_train)
pred_gbc = gbc.predict(xv_test)
print("Gradient Boosting Classifier Score:", gbc.score(xv_test, y_test))
print("Gradient Boosting Classification Report:")
print(classification_report(y_test, pred_gbc))

def output_label(n):
    if n == 0:
        return "FAKE NEWS"
    elif n == 1:
        return "GENUINE NEWS"

def manual_testing(news_article):
    testing_news = {"text": [news_article]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_xv_test = vectorization.transform(new_def_test["text"])
    
    pred_lr = LR.predict(new_xv_test)
    pred_dtc = DTC.predict(new_xv_test)
    pred_gbc = gbc.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    
    return "\n\nLR Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}\nDTC Prediction: {}".format(
        output_label(pred_lr[0]), output_label(pred_gbc[0]), output_label(pred_rfc[0]), output_label(pred_dtc[0])
    )

models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(vectorization, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
joblib.dump(LR, os.path.join(models_dir, 'logistic_regression_model.pkl'))
joblib.dump(DTC, os.path.join(models_dir, 'decision_tree_model.pkl'))
joblib.dump(rfc, os.path.join(models_dir, 'random_forest_model.pkl'))
joblib.dump(gbc, os.path.join(models_dir, 'gradient_boosting_model.pkl'))

print(f"\nModels saved to '{models_dir}/'")
