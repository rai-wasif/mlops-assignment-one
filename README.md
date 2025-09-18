# Mlops-assignment-one
This is my first mlops learning work
# MLOps Assignment 1: Introduction, GitHub Basics, and MLflow Tracking

## 📌 Objective
The purpose of this assignment is to:
- Learn GitHub for version control and collaboration.
- Train and compare multiple ML models.
- Use MLflow for experiment tracking, logging, and model registration.
- Build a reproducible ML workflow.

---

## 📂 Project Structure
mlops-assignment-1/
├── data/ # (if any datasets are stored locally)
├── notebooks/ # Jupyter notebooks (EDA, experiments)
├── src/ # Source code (training scripts)
│ ├── train_models.py
│ ├── train_with_mlflow.py
│ └── ...
├── models/ # Saved models (.pkl)
├── results/ # Confusion matrices, evaluation results
└── README.md # Documentation


---

## 📊 Dataset
- **Dataset used**: Iris dataset (from scikit-learn)
- **Samples**: 150 rows
- **Features**: Sepal length, Sepal width, Petal length, Petal width
- **Classes**: 3 flower species (`setosa`, `versicolor`, `virginica`)

---

## 🤖 Models Trained
1. **Logistic Regression**
2. **Random Forest Classifier**
3. **Support Vector Machine (SVM)**

---

## 🏋️ Training & Evaluation
- Models were trained using **train/test split (20% test)** without a fixed random state.  
- Metrics evaluated:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
- Cross-validation (5-fold) was also used for more reliable accuracy estimation.  

### 📊 Final Results

#### Random Split (single run)
| Model              | Accuracy | Precision | Recall | F1   |
|--------------------|----------|-----------|--------|------|
| LogisticRegression | 1.0000   | 1.0000    | 1.0000 | 1.0000 |
| RandomForest       | 0.9667   | 0.9697    | 0.9697 | 0.9683 |
| SVM                | 1.0000   | 1.0000    | 1.0000 | 1.0000 |

#### Cross-Validation (5-Fold Mean Accuracy)
| Model              | CV Mean Accuracy |
|--------------------|------------------|
| LogisticRegression | 0.9733           |
| RandomForest       | 0.9533           |
| SVM                | 0.9667           |

✅ Confusion matrices for each model are saved in the `/results` folder as PNGs.  
✅ Trained models are saved in the `/models` folder as `.pkl` files.  

---

## 📈 MLflow Tracking
Using `train_with_mlflow.py`, the following were logged to MLflow:
- **Parameters** (e.g., `n_estimators`, `max_iter`)
- **Metrics** (accuracy, precision, recall, F1, cross-validation accuracy)
- **Artifacts** (confusion matrices as PNGs)
- **Models** (saved in MLflow’s model registry)

### How to Run MLflow UI
```bash
mlflow ui
Open browser at: http://127.0.0.1:5000

🏆 Model Registration

All three models were successfully registered in the MLflow Model Registry.
Each has version 1 created in the registry.
Logistic Regression performed best overall (CV accuracy ~97%).

👨‍💻 Author 
Name: Muhammad Wasif
Course: MLOps Assignment 1
University: FAST-NUCES, CFD Campus
