from sklearn.model_selection import train_test_split

# імпортуємо необхідні бібліотеки для препроцесингу
import numpy as np
import pandas as pd

# імпортуємо датасет
dataset = pd.read_excel(r"phishing_website_dataset.xlsx")

# формуємо вибірки даних
data_y = dataset['Result'].values
data_x = dataset.drop(columns=['index', 'Result']).values

# Розділяємо датасет на тренувальний і тимчасовий (20% + 15% = 35%)
train_x, temp_x, train_y, temp_y = train_test_split(data_x, data_y, test_size=0.35, random_state=42)

# Розділяємо тимчасовий датасет на тестовий і валідаційний
test_size = 0.20 / 0.35  # 20% від усього датасету
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=test_size, random_state=42)

# Зберігаємо датасети у xlsx файли
pd.DataFrame(train_x).to_excel(r"train_dataset.xlsx", index=False)
pd.DataFrame(test_x).to_excel(r"test_dataset.xlsx", index=False)
pd.DataFrame(val_x).to_excel(r"validation_dataset.xlsx", index=False)
