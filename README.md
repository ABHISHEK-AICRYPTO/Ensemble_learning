# 🌸 Iris Flower Classification using Bagging Ensemble Method

This project demonstrates the implementation of an **Ensemble Learning** technique using Scikit-Learn. It utilizes a **Bagging Classifier** with a **Decision Tree** as the base estimator to classify the famous Iris flower dataset.

## 🚀 Features
- Uses the classic **Iris Dataset** (150 samples, 4 features, 3 classes).
- Implements **Bagging (Bootstrap Aggregating)** to reduce variance and prevent overfitting.
- Evaluates model performance using **Accuracy Score** on both training and testing subsets.
- Maps numerical predictions back to actual biological class names (`setosa`, `versicolor`, `virginica`).

## 📊 Model Performance
The model achieves perfect classification scores due to the clean boundary separations of the Iris dataset:
- **Training Accuracy:** 100% (`1.0`)
- **Testing Accuracy:** 100% (`1.0`)

## 🛠️ Prerequisites
To run this machine learning script or Jupyter notebook, you need Python installed along with the **scikit-learn** library. You can install it via pip:

```bash
pip install scikit-learn
```

## 💻 Code Overview & Implementation
The core script splits the dataset into an 80/20 train-test ratio and fits 10 ensemble Decision Trees:

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load and split dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Initialize Bagging Classifier
base_classifier = DecisionTreeClassifier()
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10, random_state=42)

# Train the model
bagging_classifier.fit(X_train, y_train)

# Predictions
y_pred = bagging_classifier.predict(X_test)
```

## 🔍 Sample Prediction Output
When mapping the predicted target arrays to actual species names, the model outputs:
`['versicolor', 'setosa', 'virginica', 'versicolor', ...]`
