import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

print("Step 1: Loading Dataset...")
df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Step 2: Cleaning Data...")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df.drop(columns=["customerID"], inplace=True)

target = "Churn"
X = df.drop(columns=[target])
y = df[target]

num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_features = [col for col in X.columns if col not in num_features]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_features)
    ]
)

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Step 3: Training Model...")
model_pipeline.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model_pipeline, "model/churn_model.pkl")
print("SUCCESS: Model saved to model/churn_model.pkl")
