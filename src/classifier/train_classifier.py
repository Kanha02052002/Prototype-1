import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib

def train_hybrid_model(data_path='data/filtered_data.csv', model_path='models/hybrid_classifier.pkl'):
    df = pd.read_csv(data_path)
    X = df['Summary']
    y = df['Request Type']

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)

    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('lgb', LGBMClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]

    meta_model = LogisticRegression()

    clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    clf.fit(X_vec, y)

    joblib.dump((clf, vectorizer), model_path)
    print("Hybrid classifier trained and saved.")

if __name__ == "__main__":
    train_hybrid_model()