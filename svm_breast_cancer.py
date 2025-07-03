# Support Vector Machine (SVM) Classification on Breast Cancer Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA

# Load and preprocess dataset
df = pd.read_csv('breast-cancer.csv')
df = df.loc[:, ~df.columns.str.contains('^Unnamed|id', case=False)]
X = df.drop('diagnosis', axis=1)
y = df['diagnosis'].map({'M': 1, 'B': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)
print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))

# RBF SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))

# Grid Search
param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.01, 0.1, 1]}
grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best Parameters from Grid Search:", grid.best_params_)
print("Cross-Validation Accuracy (mean):", round(cross_val_score(grid.best_estimator_, X_train_scaled, y_train, cv=5).mean(), 4))

# Classification report and confusion matrix
print("\nClassification Report (RBF Kernel):\n", classification_report(y_test, y_pred_rbf))
ConfusionMatrixDisplay.from_estimator(svm_rbf, X_test_scaled, y_test, cmap='coolwarm')
plt.title("Confusion Matrix - RBF SVM")
plt.show()

# PCA and Decision Boundary
X_combined_scaled = np.vstack((X_train_scaled, X_test_scaled))
y_combined = np.hstack((y_train, y_test))
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_combined_scaled)
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_pca.fit(X_pca, y_combined)

def plot_decision_boundary(clf, X, y, title):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k', s=30)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.show()

plot_decision_boundary(svm_pca, X_pca, y_combined, "Decision Boundary (RBF SVM, PCA 2D)")
