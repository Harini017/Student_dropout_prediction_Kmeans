# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_PATH = "student_dropout_dataset.csv"
MODEL_PATH = "model_pipeline.joblib"

# Load and clean dataset
def load_and_clean(path):
    df = pd.read_csv(path, parse_dates=['enroll_date', 'last_active_date'])
    df.fillna(0, inplace=True)
    # Feature engineering
    today = pd.Timestamp.now().normalize()
    df['days_since_last_active'] = (today - df['last_active_date']).dt.days.clip(lower=0)
    df['enrolled_days'] = (today - df['enroll_date']).dt.days.clip(lower=0)
    return df

# Build preprocessing + model pipeline
def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    return pipeline

def main():
    df = load_and_clean(DATA_PATH)

    # Exclude target columns from features
    feature_cols = [col for col in df.columns if col not in ['dropout_label', 'risk_category']]
    X = df[feature_cols]
    y = df['risk_category']

    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    pipeline = build_pipeline(numeric_features, categorical_features)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification report on test set:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model pipeline to {MODEL_PATH}")

if __name__ == "__main__":
    main()
