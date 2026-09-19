import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Use a relative import since both files are in the same 'src' folder
from .preprocessing import load_and_preprocess_data, get_preprocessing_pipeline

def train_model(data_path):
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = get_preprocessing_pipeline(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    print("Training the Random Forest model...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    joblib.dump(pipeline, 'model_pipeline.pkl')
    print("\nModel pipeline saved successfully as 'model_pipeline.pkl'!")

if __name__ == "__main__":
    train_model("data/telco_churn.csv")