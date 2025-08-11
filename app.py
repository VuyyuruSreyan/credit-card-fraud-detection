import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE  # ✅ For handling imbalance

def main():
    print("🚀 Credit Card Fraud Detection Model Started...\n")

    # Step 1: Load dataset
    file_path = Path("creditcard.csv")
    if not file_path.exists():
        print(f"[❌ ERROR] Dataset file not found at: {file_path.resolve()}")
        print("👉 Please download it from Kaggle and place it in the same folder.")
        return
    else:
        print(f"[✅] Found dataset file: {file_path}")
        df = pd.read_csv(file_path)

    # Step 2: Confirm columns
    print("[📋] Columns:", df.columns.tolist())
    required = ['Time', 'Amount', 'Class']
    for col in required:
        if col not in df.columns:
            print(f"[❌] Missing column: {col}")
            return

    # Step 3: Scale Amount & Time
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time_scaled'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    df = df.drop(['Time', 'Amount'], axis=1)

    # Step 4: Split features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Step 5: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Fraud count before SMOTE: {sum(y_train==1)}")

    # Step 6: Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Fraud count after SMOTE: {sum(y_train==1)}")

    # Step 7: Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✅ Model training complete")

    # Step 8: Predictions
    y_pred = model.predict(X_test)

    # Step 9: Evaluation
    print("\n[📊] Classification Report:\n", classification_report(y_test, y_pred))
    print("✅ Accuracy:", accuracy_score(y_test, y_pred))

    # Step 10: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Fraud (0)', 'Fraud (1)'])
    disp.plot(cmap='YlGnBu', values_format='d')
    plt.title('Confusion Matrix: Credit Card Fraud Detection')
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    main()
