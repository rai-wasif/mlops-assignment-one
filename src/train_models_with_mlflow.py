import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Define models
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}

# Create MLflow experiment
mlflow.set_experiment("Iris-Model-Comparison")

# ===== Random Split (no random_state) =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Random Split metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro')
        rec = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        # Cross-validation mean accuracy
        cv_mean_acc = cross_val_score(model, X, y, cv=5).mean()

        # Log parameters
        if name == "RandomForest":
            mlflow.log_param("n_estimators", 100)
        elif name == "LogisticRegression":
            mlflow.log_param("max_iter", 200)

        # Log metrics
        mlflow.log_metric("accuracy_split", acc)   # random split accuracy
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("cv_mean_accuracy", cv_mean_acc)  # cross-validation accuracy

        # Example input for model signature
        example = X_test[:2]

        # Log model + Register in MLflow Model Registry 
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model", 
            input_example=example,
            registered_model_name=f"Iris_{name}"
        )

        # ===== Confusion Matrix Plot =====
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=iris.target_names, yticklabels=iris.target_names)
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Save plot locally
        plot_path = f"{name}_confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()

        # Log as artifact in MLflow
        mlflow.log_artifact(plot_path, artifact_path="plots")

        # Console output
        print(f"\n{name} logged and registered in MLflow:")
        print(f"Random Split - Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
        print(f"Cross-Validation - Mean Accuracy={cv_mean_acc:.4f}")
        print(f"Confusion matrix saved and logged to MLflow: {plot_path}")