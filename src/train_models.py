import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import joblib
import os

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}

results = {}

# Folders
save_dir_models = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
save_dir_results = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(save_dir_models, exist_ok=True)
os.makedirs(save_dir_results, exist_ok=True)

# ===== Random Split (no random_state) =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='macro'),
        "recall": recall_score(y_test, y_pred, average='macro'),
        "f1": f1_score(y_test, y_pred, average='macro')
    }

    # Save model
    save_path = os.path.join(save_dir_models, f"{name}.pkl")
    joblib.dump(model, save_path)

    # Confusion Matrix (PNG)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(save_dir_results, f"{name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

print("Model Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, score in metrics.items():
        print(f"  {metric}: {score:.4f}")
print("Models saved in ../models directory")