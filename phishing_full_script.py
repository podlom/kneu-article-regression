# імпортуємо необхідні бібліотеки для препроцесингу
import numpy as np
import pandas as pd

# імпортуємо датасет
dataset = pd.read_excel(r"phishing_website_dataset.xlsx")

# формуємо вибірки даних
data_y = dataset['Result'].values
data_x = dataset.drop(columns=['index', 'Result']).values

# розбиваємо вибірки даних на тестову та трейнову
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.2, random_state=42)


# -----------------------------------------------------------------------------
# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc

log_reg = LogisticRegression(C=0.001, class_weight="balanced", n_jobs=-1)
lf = log_reg.fit(train_x, train_y)
y_pred = lf.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))
y_pred_p = lf.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y-1)))

# Отримання коефіцієнтів та їхніх стандартних помилок
#coefficients = lf.coef_[0]
#std_errors = np.std(train_x, 0) * np.std(coefficients, 0)
# Визначення статистично значущих змінних (поріг 1.96 для рівня довіри 95%)
#z_scores = coefficients / std_errors
#significant_variables = np.where(np.abs(z_scores) > 1.96)[0]

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# класифікація з масивом параметрів регуляризації
#for i in [1,0.5,0.1,0.01,0.001,0.0001,0.0000001]:
#    log_reg = LogisticRegression(C=i, class_weight="balanced", n_jobs=-1)
#    lf = log_reg.fit(train_x, train_y)
#    y_pred_p = lf.predict_proba(test_x)
#    print(i," - AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y)))
#
# класифікація з масивом алгоритмів
#for i in ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
#    log_reg = LogisticRegression(solver=i, class_weight="balanced")
#    lf = log_reg.fit(train_x, train_y)
#    y_pred_p = lf.predict_proba(test_x)
#    print(i," - AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y)))


# -----------------------------------------------------------------------------
# 2. MLP lbfgs
from sklearn.neural_network import MLPClassifier

nn_clf = MLPClassifier(solver='lbfgs', alpha=1, hidden_layer_sizes=(5, 20), random_state=1, max_iter=10000)
nn_clf.fit(train_x, train_y)
y_pred = nn_clf.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))
y_pred_p = nn_clf.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y-1)))

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# -----------------------------------------------------------------------------
# 3. MLP adam
nn_clf = MLPClassifier(solver='adam', learning_rate_init=0.00001, alpha=1, hidden_layer_sizes=(13, 3), random_state=1, max_iter=100000, verbose = True)
nn_clf.fit(train_x, train_y)
y_pred = nn_clf.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))
y_pred_p = nn_clf.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y-1)))

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# -----------------------------------------------------------------------------
# 4. Decision Tree
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=21, max_features=0.9, criterion="gini", random_state=2)
dt = tree_clf.fit(train_x, train_y)
y_pred = dt.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))
y_pred_p = dt.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y-1)))

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,10)
plt.figure()
plot_tree(dt, filled=True, class_names=['no','yes'], feature_names=dataset.select_dtypes(exclude=['object']).columns)
plt.show()


# -----------------------------------------------------------------------------
# 5. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(max_depth=4, min_samples_leaf=21, max_features=0.9, criterion="gini", n_estimators=1000, random_state=42)
rf = rf_clf.fit(train_x, train_y)
y_pred = rf.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))
y_pred_p = rf.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y-1)))

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# -----------------------------------------------------------------------------
# 6. KNN
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=8)
knn = knn_clf.fit(train_x, train_y)
y_pred = knn.predict(test_x)

print(confusion_matrix(test_y, y_pred))
print(classification_report(test_y, y_pred))
print("Accuracy: " + str(accuracy_score(test_y, y_pred)))
print("F1 score: " + str(f1_score(test_y, y_pred)))
y_pred_p = knn.predict_proba(test_x)
print("AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y-1)))

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# KNN з кількістю сусідів з 2 до 20
#for i in range(2, 21):
#    knn_clf = KNeighborsClassifier(n_neighbors = i)
#    knn = knn_clf.fit(train_x, train_y)
#    predicted_p = knn.predict_proba(test_x) 
#    predicted_p_train = knn.predict_proba(train_x) 
#    print(i, "TEST:", roc_auc_score(y_score=predicted_p[:,1], y_true=test_y), "TRAIN:",roc_auc_score(y_score=predicted_p_train[:,1], y_true=train_y))
