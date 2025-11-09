import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack
import joblib
import warnings
import logging
import os
warnings.filterwarnings('ignore')

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('model_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_hybrid_model(data_path='./data/combined_data.csv', model_path='./models/com_hybrid_classifier.pkl'):
    logger = setup_logging()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
        logger.info("Successfully loaded CSV using 'latin-1' encoding.")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(data_path, encoding='cp1252') 
            logger.info("Successfully loaded CSV using 'cp1252' encoding.")
        except UnicodeDecodeError:
            logger.error("Failed to load CSV with 'latin-1' or 'cp1252' encoding.")
            raise
    
    target_counts = df['Request Type'].value_counts()
    rare_class_threshold = 500
    classes_to_keep = target_counts[target_counts >= rare_class_threshold].index
    classes_to_group = target_counts[target_counts < rare_class_threshold].index

    df_modified = df.copy()
    df_modified.loc[df_modified['Request Type'].isin(classes_to_group), 'Request Type'] = 'Other'

    X_text = df_modified['Summary']
    y = df_modified['Request Type']

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    X_train_text = X_train_text.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    X_val_text = X_val_text.reset_index(drop=True)
    y_val = pd.Series(y_val).reset_index(drop=True)
    X_test_text = X_test_text.reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    slm = SentenceTransformer('all-MiniLM-L6-v2')

    X_train_embedded = slm.encode(X_train_text.tolist())
    X_val_embedded = slm.encode(X_val_text.tolist())
    X_test_embedded = slm.encode(X_test_text.tolist())

    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text.tolist())
    X_val_tfidf = tfidf_vectorizer.transform(X_val_text.tolist())
    X_test_tfidf = tfidf_vectorizer.transform(X_test_text.tolist())

    X_train_combined = hstack([X_train_tfidf, X_train_embedded])
    X_val_combined = hstack([X_val_tfidf, X_val_embedded])
    X_test_combined = hstack([X_test_tfidf, X_test_embedded])

    target_counts_after_split = pd.Series(y_train).value_counts()
    imbalance_threshold_percentage = 5.0
    minority_classes_combined = target_counts_after_split[target_counts_after_split / len(y_train) * 100 < imbalance_threshold_percentage]
    imbalance_detected = not minority_classes_combined.empty

    unique_train_labels, train_counts = np.unique(y_train, return_counts=True)
    classes_with_insufficient_samples_for_smote = unique_train_labels[train_counts < 2]

    X_train_to_use = X_train_combined
    y_train_to_use = y_train
    if imbalance_detected and len(classes_with_insufficient_samples_for_smote) == 0:
        min_class_samples = train_counts.min()
        k_neighbors_smote = max(1, min_class_samples - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors_smote)
        X_train_to_use, y_train_to_use = smote.fit_resample(X_train_combined, y_train)
    else:
        X_train_to_use = X_train_combined
        y_train_to_use = y_train

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train_to_use)
    X_val_scaled = scaler.transform(X_val_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    X_train_scaled_dense = X_train_scaled.toarray()
    X_val_scaled_dense = X_val_scaled.toarray()
    X_test_scaled_dense = X_test_scaled.toarray() 

    base_models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('lgb', LGBMClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]

    meta_model = LogisticRegression()

    clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    clf.fit(X_train_scaled_dense, y_train_to_use)

    y_pred_val = clf.predict(X_val_scaled_dense)
    y_pred_test = clf.predict(X_test_scaled_dense)
    
    val_accuracy = accuracy_score(y_val, y_pred_val)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")

    joblib.dump({
        'model': clf,
        'tfidf_vectorizer': tfidf_vectorizer,
        'sentence_transformer_model': 'all-MiniLM-L6-v2',
        'label_encoder': label_encoder,
        'scaler': scaler
    }, model_path)
    
    logger.info(f"Hybrid classifier trained and saved to {model_path}.")

# if __name__ == "__main__":
#     train_hybrid_model()