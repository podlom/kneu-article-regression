import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# Load dataset
dataset = pd.read_excel(r"phishing_website_dataset.xlsx")

# Data preprocessing
data_y = dataset['Result'].values
data_x = dataset.drop(columns=['index', 'Result']).values

# Standardize features
scaler = StandardScaler()
data_x_scaled = scaler.fit_transform(data_x)

# Split data
train_x, test_x, train_y, test_y = train_test_split(data_x_scaled, data_y, test_size=0.2, random_state=42)

# Define Logistic Regression with hyperparameter tuning
param_grid = {'C': [1, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001],
              'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']}
log_reg = LogisticRegression(class_weight="balanced", n_jobs=-1)
clf = GridSearchCV(log_reg, param_grid, cv=5, scoring='roc_auc')

# Fit model
clf.fit(train_x, train_y)

# Best parameters and score
print("Best parameters:", clf.best_params_)
print("Best AUC:", clf.best_score_)

# Predict and evaluate
y_pred = clf.predict(test_x)
print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))

# ROC Curve
y_pred_p = clf.predict_proba(test_x)[:,1]
fpr, tpr, _ = roc_curve(test_y, y_pred_p)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
