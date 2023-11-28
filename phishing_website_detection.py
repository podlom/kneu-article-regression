# імпорт бібліотек для препроцесингу
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
from matplotlib import pyplot as plt


# Завантаження даних - 
dataset = pd.read_excel(r"phishing_website_dataset.xlsx")

# переглядаємо датасет
dataset.shape
dataset.head()
dataset.info()

# формуємо вибірки даних
data_y = dataset['Result'].values
data_x = dataset.drop(columns=['index', 'Result']).values

# розбиваємо вибірки даних на тестову та трейнову
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size = 0.2, random_state=42)

# -----------------------------------------------------------------------------
# Logistic Regression
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
coefficients = lf.coef_[0]
std_errors = np.std(train_x, 0) * np.std(coefficients, 0)

# Визначення статистично значущих змінних (поріг 1.96 для рівня довіри 95%)
z_scores = coefficients / std_errors
significant_variables = np.where(np.abs(z_scores) > 1.96)[0]

for s in significant_variables:
    print("Significant variable index: ", s)
    print("Significant variable value: ", coefficients[s])

# ROC крива
fpr, tpr, threshold = roc_curve(test_y, y_pred_p[:,1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


#
# Solver variants
# solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
#
# класифікація з масивом параметрів регуляризації
for i in [1,0.5,0.1,0.01,0.001,0.0001,0.0000001]:
    log_reg = LogisticRegression(C=i, class_weight="balanced", n_jobs=-1)
    lf = log_reg.fit(train_x, train_y)
    y_pred_p = lf.predict_proba(test_x)
    print(i," - AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y)))


# класифікація з масивом алгоритмів
for i in ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']:
    log_reg = LogisticRegression(solver=i, class_weight="balanced")
    lf = log_reg.fit(train_x, train_y)
    y_pred_p = lf.predict_proba(test_x)
    print(i," - AUC: " + str(roc_auc_score(y_score=y_pred_p[:,1], y_true=test_y)))