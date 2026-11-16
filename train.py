"""Train and evaluate models that predict whether a student will attend regularly."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("attendance_data.csv").drop_duplicates().dropna()
X, y = df.drop(columns="attends_regularly"), df["attends_regularly"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

best_name, best_model, best_cv = None, None, 0
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    print(f"{name}: 5-fold CV accuracy = {cv:.3f}")
    if cv > best_cv:
        best_name, best_model, best_cv = name, model, cv

best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)
print(f"\nBest model: {best_name}")
print(f"Test accuracy: {accuracy_score(y_test, pred):.3f}\n")
print(classification_report(y_test, pred, target_names=["irregular", "regular"]))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))
joblib.dump(best_model, "model.joblib")
