import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("gearbox_faults.csv")  # Make sure the CSV is in the correct path

# Encode categorical variables
label_cols = ['Gear Type', 'Oil Condition']
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

# Features and target
X = df.drop("Fault Type", axis=1)
Y = df["Fault Type"]

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Polynomial Logistic Regression": make_pipeline(PolynomialFeatures(degree=2), LogisticRegression(max_iter=1000)),
    "Ridge Classifier": RidgeClassifier(),
    "Support Vector Classifier": make_pipeline(StandardScaler(), SVC()),
    "Lasso Logistic Regression": LogisticRegression(penalty='l1', solver='saga', max_iter=1000),
    "Elastic Net Logistic Regression": LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)
}

# Evaluate each model
for name, model in models.items():
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    score = accuracy_score(Y_test, Y_pred)
    print("=" * 50)
    print(f"Model: {name}")
    print(f"Accuracy Score: {score:.2f}")
    print("Classification Report:")
    print(classification_report(Y_test, Y_pred))